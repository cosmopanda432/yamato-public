"""
LoRA マージスクリプト: 学習済み LoRA アダプタを backbone に統合

train_phase2.py で生成した LoRA アダプタを backbone に組み込み、
llm-jp-eval で直接評価できる HuggingFace 形式のモデルとして保存する。

マージは CPU 上で実施 (bfloat16)。
8B モデル ≈ 16GB のシステム RAM が必要。

Usage:
    python scripts/merge_lora.py

    # カスタムチェックポイントパスを指定
    python scripts/merge_lora.py --checkpoint-dir checkpoints/phase2

    # 出力先指定
    python scripts/merge_lora.py --output-dir checkpoints/phase2/merged

その後:
    python scripts/eval_baseline.py --model-name checkpoints/phase2/merged
"""

import argparse
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA アダプタを backbone にマージして HuggingFace モデルとして保存",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "phase2"),
        help="train_phase2.py の出力ディレクトリ",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=str(REPO_ROOT / "models" / "llm-jp-4-8b-base"),
        help="ベースモデルパス (マージ元)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="マージ済みモデルの保存先 (デフォルト: checkpoint-dir/merged)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    lora_path = checkpoint_dir / "lora_adapter"
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "merged"

    # 事前チェック
    if not lora_path.exists():
        logging.error(
            "LoRA アダプタが見つかりません: %s\n"
            "先に train_phase2.py を実行してください。",
            lora_path,
        )
        return 1

    logging.info("=== LoRA マージ ===")
    logging.info("  LoRA アダプタ: %s", lora_path)
    logging.info("  ベースモデル : %s", args.base_model)
    logging.info("  出力先       : %s", output_dir)
    logging.info("注意: 8B モデルを bfloat16 でロードするため ~16GB RAM が必要です")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # 1. ベースモデルを CPU bfloat16 でロード（量子化なし）
    logging.info("ベースモデルロード中 (cpu, bfloat16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    # 2. LoRA アダプタをロード
    logging.info("LoRA アダプタをロード中...")
    model = PeftModel.from_pretrained(
        base_model,
        str(lora_path),
        torch_dtype=torch.bfloat16,
    )

    # 3. LoRA をマージして解放
    logging.info("LoRA をマージ中...")
    model = model.merge_and_unload()

    # 4. 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("マージ済みモデルを保存中: %s", output_dir)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    logging.info("=== マージ完了 ===")
    logging.info("評価コマンド:")
    logging.info(
        "  python scripts/eval_baseline.py --model-name %s "
        "--llm-jp-eval-path external/llm-jp-eval "
        "--output-dir results/phase2",
        output_dir,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
