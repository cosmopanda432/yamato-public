"""
Phase 2 学習スクリプト: yamatoLLM — QLoRA + カスタムヘッド

目的:
    カスタムヘッド (OmoikaneIntentRouter + BonpuConfidence) を追加した状態で
    ベースラインより llm-jp-eval スコアが上がるかを検証する学習を実施。

学習戦略:
    - Backbone (llm-jp-4-8b-base): 4bit 量子化 + QLoRA (RTX 3060 12GB で動作)
    - Custom heads: フル精度で学習 (bfloat16)
    - 損失: base_loss + 0.5*route_loss + 0.3*conf_loss
    - データ: llm-jp-eval tuning/train splits (jcommonsenseqa / jnli / jsquad)
    - ルートラベル: 全て 0 (chat) — codegen は保留
    - 信頼度ラベル: 1.0 (全て正解データ)

出力:
    checkpoints/phase2/
        lora_adapter/      # PEFT LoRA adapter (merge_lora.py でマージ)
        custom_heads.pt    # カスタムヘッド重み
        training_log.json  # 損失ログ

次のステップ:
    python scripts/merge_lora.py
    python scripts/eval_baseline.py --model-name checkpoints/phase2/merged

Usage:
    # 通常実行 (RTX 3060, 4bit)
    python scripts/train_phase2.py

    # 設定確認のみ (学習しない)
    python scripts/train_phase2.py --dry-run

    # サンプル数を絞った高速試験
    python scripts/train_phase2.py --max-samples 200 --max-steps 50
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "external" / "llm-jp-eval" / "dataset" / "1.4.1" / "tuning" / "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2: yamatoLLM QLoRA + カスタムヘッド学習",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=str(REPO_ROOT / "models" / "llm-jp-4-8b-base"),
        help="ベースモデルパス",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "phase2"),
        help="チェックポイント出力先",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DATASET_ROOT),
        help="tuning/train データセットディレクトリ",
    )
    parser.add_argument(
        "--target-datasets",
        type=str,
        nargs="+",
        default=["jcommonsenseqa", "jnli", "jsquad"],
        help="学習に使うデータセット（xlsum_ja は長いため省略可）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="各データセットから使用するサンプル数上限",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="最大シーケンス長（RTX 3060 メモリに合わせて設定）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="マイクロバッチサイズ",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=16,
        help="勾配累積ステップ数（effective batch size = batch_size * grad_accum）",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="LoRA の学習率",
    )
    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-3,
        help="カスタムヘッドの学習率（LoRA より高め）",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="最大ステップ数 (-1 で epoch ベース)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank (小さいほど高速・低メモリ)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj", "gate_proj"],
    )
    parser.add_argument(
        "--route-loss-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--conf-loss-weight",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="データ・設定確認のみ。学習しない。",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ============================================================
# データロード
# ============================================================

def load_training_data(
    dataset_dir: Path,
    target_datasets: List[str],
    max_samples: int,
    seed: int,
) -> List[Dict]:
    """llm-jp-eval tuning/train から学習データをロード"""
    all_samples = []
    rng = random.Random(seed)

    for name in target_datasets:
        path = dataset_dir / f"{name}.json"
        if not path.exists():
            logging.warning("データセットが見つかりません: %s", path)
            continue

        with open(path, encoding="utf-8") as f:
            samples = json.load(f)

        if len(samples) > max_samples:
            samples = rng.sample(samples, max_samples)

        for s in samples:
            s["_dataset"] = name

        all_samples.extend(samples)
        logging.info("  %s: %d samples", name, len(samples))

    rng.shuffle(all_samples)
    logging.info("合計学習サンプル数: %d", len(all_samples))
    return all_samples


# ============================================================
# トークナイズ / 損失マスク
# ============================================================

def tokenize_with_loss_mask(
    sample: Dict,
    tokenizer,
    max_length: int,
) -> Optional[Dict]:
    """
    text を tokenize し、応答部分のみ loss を計算する labels を生成。

    "### 応答:\n" 以降のみ labels を残し、それ以前は -100 でマスク。
    """
    import torch

    text = sample["text"]
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"][0]
    attention_mask = enc["attention_mask"][0]

    # 応答部分のみ loss 計算
    response_marker = "### 応答:\n"
    marker_ids = tokenizer.encode(response_marker, add_special_tokens=False)
    marker_len = len(marker_ids)

    labels = input_ids.clone()
    # デフォルト: 全マスク
    labels[:] = -100

    # marker の開始位置を探す
    seq_len = len(input_ids)
    found = False
    for i in range(seq_len - marker_len + 1):
        if input_ids[i : i + marker_len].tolist() == marker_ids:
            # marker の直後から loss 計算
            labels[i + marker_len :] = input_ids[i + marker_len :]
            found = True
            break

    if not found:
        # マーカーが見つからない場合は全体を学習対象に
        labels = input_ids.clone()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collate_fn(batch: List[Dict], pad_token_id: int):
    """可変長シーケンスをパディングしてバッチ化"""
    import torch

    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids_list.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id)])
        )
        attention_mask_list.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100)])
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }


# ============================================================
# モデル構築
# ============================================================

def build_model(args: argparse.Namespace):
    """
    4bit 量子化 backbone + LoRA + カスタムヘッドを構築

    Returns:
        yamato_model: YamatoLLM インスタンス
        lora_params: LoRA パラメータグループ (optimizer 用)
        head_params: カスタムヘッドパラメータグループ (optimizer 用)
    """
    import torch
    import sys
    sys.path.insert(0, str(REPO_ROOT))

    from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    from kojiki_lm.yamato_model import YamatoLLM
    from kojiki_lm.yamato_config import YamatoConfig, LoRAConfig
    from kojiki_lm.qwen_adapter import QwenAdapter

    logging.info("Backbone ロード中 (4bit): %s", args.model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # QLoRA の準備: prepare_model_for_kbit_training は PEFT 0.5.x で全パラメータを
    # float32 にキャストしようとして OOM するため、直接 gradient checkpointing を設定する
    backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    else:
        # transformers < 4.35 向けフォールバック
        def _make_inputs_require_grad(module, inp, out):
            out.requires_grad_(True)
        backbone.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

    # LoRA 注入
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=args.lora_target_modules,
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    backbone.print_trainable_parameters()

    # YamatoLLM ラッパー構築
    yamato_config = YamatoConfig()
    yamato_config.backbone.model_name = args.model_name

    model = YamatoLLM(backbone=backbone, tokenizer=tokenizer, config=yamato_config)
    model.init_custom_heads()

    # パラメータグループ分離（異なる学習率を設定するため）
    lora_params = [p for n, p in backbone.named_parameters() if p.requires_grad]
    head_params = list(model.custom_heads.parameters())

    trainable_lora = sum(p.numel() for p in lora_params)
    trainable_heads = sum(p.numel() for p in head_params)
    logging.info(
        "学習パラメータ: LoRA %.2fM + CustomHeads %.2fM = 合計 %.2fM",
        trainable_lora / 1e6, trainable_heads / 1e6,
        (trainable_lora + trainable_heads) / 1e6,
    )

    return model, lora_params, head_params, tokenizer


# ============================================================
# 学習ループ
# ============================================================

def train(args: argparse.Namespace, model, lora_params, head_params, tokenizer, samples: List[Dict]):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    device = next(model.backbone.parameters()).device

    # カスタムヘッドをモデルのデバイスに合わせる
    model.custom_heads = model.custom_heads.to(device)

    class TextDataset(Dataset):
        def __init__(self, samples, tokenizer, max_length):
            self.items = []
            skipped = 0
            for s in samples:
                encoded = tokenize_with_loss_mask(s, tokenizer, max_length)
                if encoded is not None:
                    self.items.append(encoded)
                else:
                    skipped += 1
            if skipped:
                logging.warning("トークナイズスキップ: %d サンプル", skipped)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    dataset = TextDataset(samples, tokenizer, args.max_seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=0,
    )
    logging.info("データローダー: %d バッチ/エポック", len(dataloader))

    try:
        # bitsandbytes paged AdamW: optimizer state を CPU に退避してVRAM節約
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW([
            {"params": lora_params, "lr": args.learning_rate},
            {"params": head_params, "lr": args.head_lr},
        ], weight_decay=0.01)
        logging.info("Optimizer: bitsandbytes PagedAdamW (VRAM節約)")
    except (ImportError, AttributeError):
        optimizer = torch.optim.AdamW([
            {"params": lora_params, "lr": args.learning_rate},
            {"params": head_params, "lr": args.head_lr},
        ], weight_decay=0.01)
        logging.info("Optimizer: PyTorch AdamW")

    # ルートラベル: 全て chat (0)
    # 信頼度ラベル: 1.0 (正解データのみ)
    route_label_val = 0  # ROUTE_CHAT
    conf_label_val = 1.0

    total_steps = 0
    log_entries = []

    model.backbone.train()
    model.custom_heads.train()

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        grad_accum_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(dataloader):
            if args.max_steps > 0 and total_steps >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            bsz = input_ids.shape[0]

            route_labels = torch.full(
                (bsz,), route_label_val, dtype=torch.long, device=device
            )
            confidence_labels = torch.full(
                (bsz,), conf_label_val, dtype=torch.float, device=device
            )

            # autocast: backbone は bfloat16、カスタムヘッドも統一して bfloat16 で計算
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    route_labels=route_labels,
                    confidence_labels=confidence_labels,
                )

            loss = out["loss"] / args.grad_accum
            loss.backward()
            grad_accum_loss += loss.item()

            is_update_step = (step + 1) % args.grad_accum == 0
            is_last_step = step == len(dataloader) - 1

            if is_update_step or is_last_step:
                torch.nn.utils.clip_grad_norm_(
                    lora_params + list(model.custom_heads.parameters()), 1.0
                )
                optimizer.step()
                optimizer.zero_grad()

                total_steps += 1
                epoch_loss += grad_accum_loss

                if total_steps % 10 == 0 or total_steps <= 5:
                    base_l = out.get("base_loss", torch.tensor(0.0)).item()
                    route_l = out.get("route_loss", torch.tensor(0.0)).item()
                    conf_l = out.get("confidence_loss", torch.tensor(0.0)).item()
                    logging.info(
                        "Epoch %d | Step %d | loss=%.4f (base=%.4f route=%.4f conf=%.4f)",
                        epoch + 1, total_steps,
                        grad_accum_loss * args.grad_accum,
                        base_l, route_l, conf_l,
                    )
                    log_entries.append({
                        "epoch": epoch + 1,
                        "step": total_steps,
                        "loss": grad_accum_loss * args.grad_accum,
                        "base_loss": base_l,
                        "route_loss": route_l,
                        "conf_loss": conf_l,
                    })

                grad_accum_loss = 0.0

        avg_loss = epoch_loss / max(1, len(dataloader) // args.grad_accum)
        logging.info("Epoch %d 完了 | avg_loss=%.4f", epoch + 1, avg_loss)

        if args.max_steps > 0 and total_steps >= args.max_steps:
            logging.info("--max-steps %d に達したため学習終了", args.max_steps)
            break

    return log_entries


# ============================================================
# チェックポイント保存
# ============================================================

def save_checkpoint(model, output_dir: Path, log_entries: List[Dict]) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    # LoRA adapter
    lora_path = output_dir / "lora_adapter"
    model.backbone.save_pretrained(str(lora_path))
    logging.info("LoRA アダプタ保存: %s", lora_path)

    # カスタムヘッド
    head_path = output_dir / "custom_heads.pt"
    torch.save(model.custom_heads.state_dict(), str(head_path))
    logging.info("カスタムヘッド保存: %s", head_path)

    # 学習ログ
    log_path = output_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, ensure_ascii=False, indent=2)
    logging.info("学習ログ保存: %s", log_path)


# ============================================================
# メイン
# ============================================================

def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import random as _random
    _random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    output_dir = Path(args.output_dir)
    dataset_dir = Path(args.dataset_dir)

    logging.info("=== Phase 2 学習設定 ===")
    logging.info("  モデル    : %s", args.model_name)
    logging.info("  データセット: %s", args.target_datasets)
    logging.info("  max_samples: %d / dataset", args.max_samples)
    logging.info("  epochs    : %d", args.num_epochs)
    logging.info("  batch     : %d (accum=%d, effective=%d)",
                 args.batch_size, args.grad_accum, args.batch_size * args.grad_accum)
    logging.info("  LoRA r=%d alpha=%d lr=%.1e", args.lora_r, args.lora_alpha, args.learning_rate)
    logging.info("  head lr   : %.1e", args.head_lr)
    logging.info("  出力先    : %s", output_dir)

    # データロード
    samples = load_training_data(
        dataset_dir=dataset_dir,
        target_datasets=args.target_datasets,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    if not samples:
        logging.error("学習データが空です。--dataset-dir を確認してください: %s", dataset_dir)
        return 1

    if args.dry_run:
        logging.info("--dry-run: 学習はスキップします")
        logging.info("サンプル例: %s", json.dumps(samples[0], ensure_ascii=False)[:200])
        return 0

    # モデル構築
    try:
        model, lora_params, head_params, tokenizer = build_model(args)
    except Exception as e:
        logging.exception("モデル構築失敗: %s", e)
        return 1

    # 学習
    t0 = time.time()
    try:
        log_entries = train(args, model, lora_params, head_params, tokenizer, samples)
    except Exception as e:
        logging.exception("学習失敗: %s", e)
        return 1
    elapsed = time.time() - t0
    logging.info("学習時間: %.1f 秒 (%.1f 分)", elapsed, elapsed / 60)

    # 保存
    save_checkpoint(model, output_dir, log_entries)

    logging.info("=== Phase 2 学習完了 ===")
    logging.info("次のステップ:")
    logging.info("  python scripts/merge_lora.py --checkpoint-dir %s", output_dir)
    logging.info("  python scripts/eval_baseline.py --model-name %s/merged", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
