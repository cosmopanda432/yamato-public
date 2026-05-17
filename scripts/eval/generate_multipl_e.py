"""
MultiPL-E TS の prompt を読み込み、Qwen2.5-Coder-7B-Instruct で
completion を生成する。

各サンプルを JSON で保存:
    {name, sample_id, prompt, completion, tests, stop_tokens, raw_completion}

Pass@1 評価は run_tests.py に分離。

使い方:
    python3 scripts/eval/generate_multipl_e.py \
        --input data/raw/multipl_e/humaneval-ts/test-00000-of-00001.parquet \
        --out-dir data/eval/generated/humaneval-ts \
        --quantize 4bit \
        --limit 5  # まずは 5問でパイロット
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from kojiki_lm.qwen_adapter import QwenAdapter


def truncate_at_stop_tokens(text: str, stop_tokens) -> str:
    cut = len(text)
    for st in stop_tokens or []:
        idx = text.find(st)
        if idx >= 0 and idx < cut:
            cut = idx
    return text[:cut]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--quantize", default="4bit", choices=["4bit", "8bit", "none"])
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None,
                    help="最初の N 問のみ処理（パイロット用）")
    ap.add_argument("--skip-existing", action="store_true",
                    help="出力ファイルが既存ならスキップ")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    quantize = None if args.quantize == "none" else args.quantize

    print(f"Loading {args.model} (quantize={quantize})")
    t0 = time.time()
    model, tokenizer = QwenAdapter.load_base_model(
        model_name=args.model,
        quantize=quantize,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    print(f"Loading prompts from {args.input}")
    table = pq.read_table(args.input)
    rows = table.to_pylist()
    if args.limit is not None:
        rows = rows[: args.limit]
    print(f"  {len(rows)} problems")

    sample_id = 0
    n_done = 0
    n_skipped = 0
    t_total = 0.0
    for i, row in enumerate(rows):
        name = row["name"]
        out_path = out_dir / f"{name}__s{sample_id}.json"
        if args.skip_existing and out_path.exists():
            n_skipped += 1
            continue

        prompt = row["prompt"]
        stop_tokens = list(row.get("stop_tokens") or [])

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t_gen = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - t_gen
        t_total += elapsed

        new_ids = outputs[0, inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True)
        completion = truncate_at_stop_tokens(raw, stop_tokens)

        out_path.write_text(json.dumps({
            "name": name,
            "sample_id": sample_id,
            "prompt": prompt,
            "completion": completion,
            "raw_completion": raw,
            "tests": row["tests"],
            "stop_tokens": stop_tokens,
            "model": args.model,
            "quantize": args.quantize,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "elapsed_sec": elapsed,
        }, ensure_ascii=False))
        n_done += 1
        print(
            f"  [{i+1}/{len(rows)}] {name}: "
            f"{len(raw)}c raw / {len(completion)}c kept, {elapsed:.1f}s"
        )

    avg = t_total / max(n_done, 1)
    print(
        f"\nDone: generated={n_done}, skipped={n_skipped}, "
        f"avg={avg:.1f}s/problem, total={t_total:.0f}s"
    )


if __name__ == "__main__":
    main()
