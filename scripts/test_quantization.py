"""
Phase 1 Task 2: 量子化テスト — TensonKorinQuantizer で INT4 変換し RTX 3060 で推論確認

ROADMAP Phase 1 Task 2:
    TensonKorinQuantizer で INT4 変換 → RTX 3060 で推論確認

完了条件:
    INT4 量子化済み llm-jp-4-8b が RTX 3060 (12GB VRAM) でテキスト生成できること。

Usage:
    # BitsAndBytes 4bit 量子化でロード → 生成テスト
    python scripts/test_quantization.py

    # 保存付き
    python scripts/test_quantization.py --output-path checkpoints/qwen35_9b_nf4/

    # ベンチマーク（複数プロンプトで latency / tokens-per-sec を計測）
    python scripts/test_quantization.py --benchmark
"""

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kojiki_lm.tenson_korin_quantizer import (
    TensonKorinConfig,
    TensonKorinQuantizer,
)


DEFAULT_PROMPTS = [
    "こんにちは。あなたは誰ですか？",
    "日本の首都はどこですか？",
    "Pythonでフィボナッチ数列を書いてください。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 Task 2: INT4 量子化推論テスト",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="llm-jp/llm-jp-4-8b-base",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="量子化済みモデルの保存先（None なら保存しない）",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="単発プロンプト（--benchmark と排他）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        choices=["nf4", "fp4"],
        default="nf4",
    )
    parser.add_argument(
        "--double-quant",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-double-quant",
        dest="double_quant",
        action="store_false",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="複数プロンプトで latency / tokens-per-sec を計測",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="ベンチマーク前のウォームアップ実行回数",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def report_environment() -> None:
    import torch

    logging.info("=" * 60)
    logging.info("環境情報")
    logging.info("=" * 60)
    logging.info("PyTorch: %s", torch.__version__)
    logging.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logging.info("CUDA device: %s", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        logging.info("VRAM: %.1f GB", props.total_memory / 1024**3)


def descend(args: argparse.Namespace):
    """天孫降臨: BitsAndBytes 4bit で量子化ロード"""
    config = TensonKorinConfig(
        quant_bits=4,
        quant_type=args.quant_type,
        double_quant=args.double_quant,
    )
    quantizer = TensonKorinQuantizer(config=config)

    logging.info("=" * 60)
    logging.info("天孫降臨 (Tenson Korin) 開始")
    logging.info("  model: %s", args.model_name)
    logging.info("  quant_type: %s, double_quant: %s",
                 args.quant_type, args.double_quant)
    logging.info("=" * 60)

    t0 = time.perf_counter()
    model, tokenizer = quantizer.descend_bnb(
        model_name=args.model_name,
        output_path=args.output_path,
    )
    elapsed = time.perf_counter() - t0
    logging.info("量子化ロード完了 (%.1fs)", elapsed)

    return model, tokenizer, quantizer


def report_vram(model, quantizer: TensonKorinQuantizer) -> None:
    import torch

    estimate = quantizer.amenouzume.estimate_vram(model)
    logging.info("推定 VRAM:")
    for k, v in estimate.items():
        logging.info("  %s: %.2f GB", k, v)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info("実測 VRAM: allocated=%.2f GB, reserved=%.2f GB",
                     allocated, reserved)


def generate_once(model, tokenizer, prompt: str, max_new_tokens: int):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    new_ids = out_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    n_new = new_ids.shape[0]
    return text, n_new, elapsed


def run_single(model, tokenizer, prompt: str, max_new_tokens: int) -> None:
    text, n_new, elapsed = generate_once(model, tokenizer, prompt, max_new_tokens)
    tps = n_new / elapsed if elapsed > 0 else 0.0

    logging.info("=" * 60)
    logging.info("生成完了: %d tokens in %.2fs (%.1f tok/s)",
                 n_new, elapsed, tps)
    logging.info("=" * 60)

    print(f"\n----- prompt -----\n{prompt}")
    print(f"\n----- 生成テキスト -----\n{text}\n")


def run_benchmark(model, tokenizer, max_new_tokens: int, warmup: int) -> None:
    # ウォームアップ
    for i in range(warmup):
        logging.info("ウォームアップ %d/%d", i + 1, warmup)
        generate_once(model, tokenizer, DEFAULT_PROMPTS[0], max_new_tokens)

    results = []
    for prompt in DEFAULT_PROMPTS:
        text, n_new, elapsed = generate_once(
            model, tokenizer, prompt, max_new_tokens
        )
        tps = n_new / elapsed if elapsed > 0 else 0.0
        results.append((prompt, n_new, elapsed, tps))
        logging.info(
            "[%s...] %d tokens / %.2fs / %.1f tok/s",
            prompt[:20], n_new, elapsed, tps,
        )

    avg_tps = sum(r[3] for r in results) / len(results)
    logging.info("=" * 60)
    logging.info("ベンチマーク結果: 平均 %.1f tok/s (%d prompts)",
                 avg_tps, len(results))
    logging.info("=" * 60)


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.prompt and args.benchmark:
        logging.error("--prompt と --benchmark は同時に指定できません")
        return 2

    try:
        report_environment()
        model, tokenizer, quantizer = descend(args)
        report_vram(model, quantizer)

        if args.benchmark:
            run_benchmark(
                model, tokenizer,
                max_new_tokens=args.max_new_tokens,
                warmup=args.warmup,
            )
        else:
            prompt = args.prompt or DEFAULT_PROMPTS[0]
            run_single(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
            )
    except Exception as exc:
        logging.exception("量子化テスト失敗: %s", exc)
        return 1

    logging.info("Phase 1 Task 2 完了 — 量子化テスト成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
