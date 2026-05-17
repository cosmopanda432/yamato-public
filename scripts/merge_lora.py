"""
LoRA マージ — 学習済み LoRA adapter を Qwen2.5-Coder backbone に統合する。

scripts/train/sft_yamato.py の出力 (lora_adapter/) を backbone にマージし、
HuggingFace 形式のスタンドアロンモデルとして保存する。custom_heads.pt は
そのまま output_dir にコピー (推論時に YamatoLLM が読み込む)。

マージは CPU bfloat16 で実施 (7B モデル ≈ 14GB の RAM が必要)。

Usage:
    python scripts/merge_lora.py \
        --checkpoint-dir checkpoints/yamato_sft/final \
        --base-model models/Qwen2.5-Coder-7B-Instruct \
        --output-dir checkpoints/yamato_sft/merged
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint-dir",
        default=str(REPO_ROOT / "checkpoints" / "yamato_sft" / "final"),
    )
    p.add_argument(
        "--base-model",
        default=str(REPO_ROOT / "models" / "Qwen2.5-Coder-7B-Instruct"),
    )
    p.add_argument("--output-dir", default=None)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    ckpt = Path(args.checkpoint_dir)
    lora_path = ckpt / "lora_adapter"
    heads_path = ckpt / "custom_heads.pt"
    out_dir = Path(args.output_dir) if args.output_dir else ckpt.parent / "merged"

    if not lora_path.exists():
        logging.error("LoRA adapter not found: %s", lora_path)
        return 1

    logging.info("=== LoRA merge ===")
    logging.info("  adapter: %s", lora_path)
    logging.info("  base:    %s", args.base_model)
    logging.info("  output:  %s", out_dir)
    logging.info("Note: Qwen2.5-Coder-7B in bfloat16 needs ~14GB RAM on CPU")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logging.info("Loading base model (cpu, bfloat16)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    logging.info("Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base, str(lora_path), torch_dtype=torch.bfloat16)

    logging.info("Merging LoRA -> base...")
    model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Saving merged model to %s ...", out_dir)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    if heads_path.exists():
        shutil.copy(heads_path, out_dir / "custom_heads.pt")
        logging.info("Copied %s -> %s", heads_path, out_dir / "custom_heads.pt")

    logging.info("=== Done ===")
    logging.info("Evaluate with:")
    logging.info(
        "  python scripts/eval/generate_multipl_e.py "
        "--input data/raw/multipl_e/humaneval-ts/test-00000-of-00001.parquet "
        "--out-dir data/eval/generated/humaneval-ts-yamato "
        "--model %s",
        out_dir,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
