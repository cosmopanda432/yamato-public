"""
yamatoLLM SFT — Qwen2.5-Coder-7B + LoRA + TsukuyomiTypeHead + BonpuConfidence。

入力:
    data/processed/sft/*.parquet
      列: input_ids, attention_mask, labels, type_labels

損失:
    base_loss (Qwen の CLM)
  + type_head.loss_weight * type_loss  (TsukuyomiTypeHead, ManyTypes4TS 由来 labels)
  + 0.3 * conf_loss (BonpuConfidence, ダミー 1.0 で安定化のみ)

出力:
    checkpoints/yamato_sft/
        lora_adapter/          # PEFT LoRA adapter
        custom_heads.pt         # type_head + confidence state_dict
        training_log.json
        config.json             # 学習設定スナップショット

使い方:
    # RTX 3060 パイロット
    python3 scripts/train/sft_yamato.py \
        --train-parquet data/processed/sft/test_mini.parquet \
        --output-dir checkpoints/yamato_sft_pilot \
        --max-steps 200 \
        --batch-size 1 --grad-accum 8 --max-seq-length 1024

    # RunPod 本番 (A100)
    python3 scripts/train/sft_yamato.py \
        --train-parquet data/processed/sft/train.parquet \
        --output-dir checkpoints/yamato_sft_full \
        --num-epochs 1 --batch-size 4 --grad-accum 4 --max-seq-length 2048 \
        --lora-r 32
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAD_LABEL = -100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--val-parquet", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default=str(REPO_ROOT / "models" / "Qwen2.5-Coder-7B-Instruct"))

    # データ
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None)

    # 学習
    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # LoRA
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-target-modules", nargs="+", default=["q_proj", "v_proj", "gate_proj"])

    # 損失
    p.add_argument("--type-loss-weight", type=float, default=0.3)
    p.add_argument("--conf-loss-weight", type=float, default=0.3)

    # 量子化
    p.add_argument("--quantize", choices=["4bit", "8bit", "none"], default="4bit")

    # ログ
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


class ParquetSFTDataset(Dataset):
    def __init__(self, parquet_path: str, limit: int = None, max_seq_length: int = None):
        table = pq.read_table(parquet_path)
        self.rows = table.to_pylist()
        if limit is not None:
            self.rows = self.rows[:limit]
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        ids = r["input_ids"]
        am = r["attention_mask"]
        lb = r["labels"]
        tl = r["type_labels"]
        if self.max_seq_length is not None and len(ids) > self.max_seq_length:
            ids = ids[: self.max_seq_length]
            am = am[: self.max_seq_length]
            lb = lb[: self.max_seq_length]
            tl = tl[: self.max_seq_length]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(am, dtype=torch.long),
            "labels": torch.tensor(lb, dtype=torch.long),
            "type_labels": torch.tensor(tl, dtype=torch.long),
        }


def collate(batch: List[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)

    def pad(t, pad_val):
        n = max_len - t.shape[0]
        if n == 0:
            return t
        return torch.cat([t, torch.full((n,), pad_val, dtype=t.dtype)])

    return {
        "input_ids": torch.stack([pad(b["input_ids"], pad_id) for b in batch]),
        "attention_mask": torch.stack([pad(b["attention_mask"], 0) for b in batch]),
        "labels": torch.stack([pad(b["labels"], PAD_LABEL) for b in batch]),
        "type_labels": torch.stack([pad(b["type_labels"], PAD_LABEL) for b in batch]),
    }


def build_model(args: argparse.Namespace):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType

    from kojiki_lm.yamato_config import YamatoConfig
    from kojiki_lm.yamato_model import YamatoLLM
    from kojiki_lm.qwen_adapter import QwenAdapter

    resolved = QwenAdapter.resolve_model_path(args.model_name)
    logging.info("Loading backbone: %s (quantize=%s)", resolved, args.quantize)

    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if args.quantize == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.quantize == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    backbone = AutoModelForCausalLM.from_pretrained(resolved, **load_kwargs)

    # QLoRA 準備（prepare_model_for_kbit_training を使わずに gradient_checkpointing 直接設定）
    backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()

    # LoRA 注入
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=args.lora_target_modules,
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_cfg)
    backbone.print_trainable_parameters()

    yamato_config = YamatoConfig()
    yamato_config.backbone.model_name = args.model_name
    yamato_config.type_head.loss_weight = args.type_loss_weight

    model = YamatoLLM(backbone=backbone, tokenizer=tokenizer, config=yamato_config)
    model.init_custom_heads()

    lora_params = [p for n, p in backbone.named_parameters() if p.requires_grad]
    head_params = list(model.custom_heads.parameters())
    n_lora = sum(p.numel() for p in lora_params)
    n_head = sum(p.numel() for p in head_params)
    logging.info(
        "Trainable: LoRA %.2fM + Heads %.2fM = %.2fM",
        n_lora / 1e6, n_head / 1e6, (n_lora + n_head) / 1e6,
    )

    return model, lora_params, head_params, tokenizer


def build_optimizer(args, lora_params, head_params):
    groups = [
        {"params": lora_params, "lr": args.learning_rate},
        {"params": head_params, "lr": args.head_lr},
    ]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW(groups, weight_decay=args.weight_decay)
        logging.info("Optimizer: bitsandbytes PagedAdamW")
    except (ImportError, AttributeError):
        opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        logging.info("Optimizer: torch AdamW")
    return opt


def main():
    args = parse_args()
    # ログは stdout に出して flush を即時に (tee/nohup 越しでも見えるように)
    import sys
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    # stdout の line buffering を強制
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    train_ds = ParquetSFTDataset(args.train_parquet, args.limit, args.max_seq_length)
    val_ds = (
        ParquetSFTDataset(args.val_parquet, max_seq_length=args.max_seq_length)
        if args.val_parquet else None
    )
    logging.info("Train: %d samples", len(train_ds))
    if val_ds:
        logging.info("Val:   %d samples", len(val_ds))

    model, lora_params, head_params, tokenizer = build_model(args)

    if args.dry_run:
        logging.info("Dry-run mode: skip training")
        return

    device = next(model.backbone.parameters()).device

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
        num_workers=0,
    )
    steps_per_epoch = len(train_loader)
    total_update_steps = max(
        (steps_per_epoch * args.num_epochs) // args.grad_accum, 1
    )
    if args.max_steps > 0:
        total_update_steps = min(total_update_steps, args.max_steps)

    optimizer = build_optimizer(args, lora_params, head_params)

    # 線形 warmup + cosine decay
    from torch.optim.lr_scheduler import LambdaLR
    import math
    warmup_steps = max(int(total_update_steps * args.warmup_ratio), 1)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_update_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    scheduler = LambdaLR(optimizer, lr_lambda)

    log_entries: List[Dict] = []
    model.backbone.train()
    model.custom_heads.train()

    total_step = 0  # update step (after grad accum)
    micro_step = 0
    accum_total = 0.0
    accum_base = 0.0
    accum_type = 0.0
    accum_conf = 0.0
    accum_count = 0
    t_start = time.time()

    optimizer.zero_grad()

    for epoch in range(args.num_epochs):
        for batch in train_loader:
            if args.max_steps > 0 and total_step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            type_labels = batch["type_labels"].to(device)
            bsz = input_ids.size(0)
            conf_labels = torch.ones(bsz, dtype=torch.float, device=device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    type_labels=type_labels,
                    confidence_labels=conf_labels,
                )

            loss = out["loss"] / args.grad_accum
            loss.backward()

            accum_total += out["loss"].item()
            accum_base += out["base_loss"].item() if torch.is_tensor(out["base_loss"]) else 0.0
            accum_type += out.get("type_loss", torch.tensor(0.0)).item() if "type_loss" in out else 0.0
            accum_conf += out.get("confidence_loss", torch.tensor(0.0)).item() if "confidence_loss" in out else 0.0
            accum_count += 1
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params + head_params, args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_step += 1

                if total_step % args.log_every == 0 or total_step == 1:
                    el = time.time() - t_start
                    rate = accum_count / max(el, 1e-6)
                    entry = {
                        "step": total_step,
                        "epoch": epoch,
                        "loss": accum_total / accum_count,
                        "base_loss": accum_base / accum_count,
                        "type_loss": accum_type / accum_count,
                        "conf_loss": accum_conf / accum_count,
                        "lr_lora": scheduler.get_last_lr()[0],
                        "lr_head": scheduler.get_last_lr()[1],
                        "micro_step": micro_step,
                        "elapsed_sec": el,
                        "samples_per_sec": rate,
                    }
                    log_entries.append(entry)
                    logging.info(
                        "step=%d total=%.4f base=%.4f type=%.4f conf=%.4f lr_lora=%.2e samples/s=%.2f",
                        total_step, entry["loss"], entry["base_loss"], entry["type_loss"],
                        entry["conf_loss"], entry["lr_lora"], entry["samples_per_sec"],
                    )
                    accum_total = accum_base = accum_type = accum_conf = 0.0
                    accum_count = 0
                    t_start = time.time()

                if total_step % args.save_every == 0:
                    save_checkpoint(out_dir, model, log_entries, args, step=total_step)

        if args.max_steps > 0 and total_step >= args.max_steps:
            break

    save_checkpoint(out_dir, model, log_entries, args, step=total_step, final=True)
    logging.info("Training done. Total update steps: %d", total_step)


def save_checkpoint(out_dir: Path, model, log_entries, args, step: int, final: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "final" if final else f"step_{step}"
    sub = out_dir / tag
    sub.mkdir(parents=True, exist_ok=True)

    # LoRA adapter
    model.backbone.save_pretrained(sub / "lora_adapter")
    # カスタムヘッド
    torch.save(model.custom_heads.state_dict(), sub / "custom_heads.pt")
    # ログ
    (sub / "training_log.json").write_text(json.dumps(log_entries, indent=2))

    logging.info("Saved checkpoint -> %s", sub)


if __name__ == "__main__":
    main()
