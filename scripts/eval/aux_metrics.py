"""
ROADMAP の補助メトリクスを集計する:
    - tsc strict pass rate    (prompt + completion を tsc --strict で再コンパイル)
    - any usage rate          (completion に `: any` / `as any` / `<any>` などが含まれる率)

tsc_strict_runner.js (Node) にパイプして判定。

使い方:
    python3 scripts/eval/aux_metrics.py \
        --generated-dir data/eval/generated/humaneval-ts \
        --out data/eval/results/humaneval-ts/_aux_metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

TSC_RUNNER = "scripts/ts_tools/dist/tsc_strict_runner.js"

# completion 中の any 使用パターン
ANY_PATTERNS = [
    re.compile(r":\s*any\b"),         # 型注釈 :any
    re.compile(r":\s*any\s*\["),      # any[]
    re.compile(r"as\s+any\b"),        # cast: as any
    re.compile(r"<any>"),             # cast: <any>
    re.compile(r"\bArray\s*<\s*any\s*>"),
    re.compile(r"\bRecord\s*<\s*[^,]+,\s*any\s*>"),
]


def count_any(text: str) -> int:
    return sum(len(p.findall(text)) for p in ANY_PATTERNS)


def run_tsc_strict_batch(samples: list[dict]) -> list[dict]:
    """tsc_strict_runner に JSONL を流して結果を JSONL で受け取る"""
    payload_lines = []
    for s in samples:
        full_code = s["prompt"] + s["completion"]
        payload_lines.append(json.dumps({"id": s["name"], "code": full_code}))
    payload = "\n".join(payload_lines) + "\n"
    proc = subprocess.run(
        ["node", TSC_RUNNER],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    results = []
    for line in proc.stdout.strip().split("\n"):
        if line:
            results.append(json.loads(line))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not Path(TSC_RUNNER).exists():
        raise SystemExit(f"{TSC_RUNNER} not found. Run `npm run build` in scripts/ts_tools.")

    gen_dir = Path(args.generated_dir)
    files = sorted(gen_dir.glob("*.json"))
    print(f"Loading {len(files)} samples from {gen_dir}")

    samples = []
    for jf in files:
        d = json.loads(jf.read_text())
        samples.append({
            "name": d["name"],
            "prompt": d["prompt"],
            "completion": d["completion"],
        })

    print("Running tsc --strict on full (prompt + completion) ...")
    tsc_results = run_tsc_strict_batch(samples)
    tsc_by_name = {r["id"]: r for r in tsc_results}

    # 集計
    per_sample = []
    n_tsc_pass = 0
    n_any_users = 0
    total_any_count = 0
    err_code_counts: dict = {}

    for s in samples:
        tsc = tsc_by_name.get(s["name"], {})
        ok = bool(tsc.get("ok", False))
        if ok:
            n_tsc_pass += 1
        any_n = count_any(s["completion"])
        total_any_count += any_n
        if any_n > 0:
            n_any_users += 1
        # error code 集計（複数同コードは1回カウント、TS◯◯◯◯ → ◯◯◯◯）
        for code in set(tsc.get("error_codes", []) or []):
            err_code_counts[code] = err_code_counts.get(code, 0) + 1

        per_sample.append({
            "name": s["name"],
            "tsc_ok": ok,
            "n_diagnostics": tsc.get("n_diagnostics", -1),
            "error_codes": tsc.get("error_codes", []),
            "any_count": any_n,
        })

    n = len(samples)
    summary = {
        "n_total": n,
        "tsc_strict_pass_rate": n_tsc_pass / max(n, 1),
        "any_usage_rate": n_any_users / max(n, 1),
        "avg_any_per_completion": total_any_count / max(n, 1),
        "top_error_codes": sorted(
            err_code_counts.items(), key=lambda x: -x[1]
        )[:10],
        "per_sample": per_sample,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\ntsc --strict pass rate = {n_tsc_pass}/{n} = {summary['tsc_strict_pass_rate'] * 100:.1f}%")
    print(f"any usage rate         = {n_any_users}/{n} = {summary['any_usage_rate'] * 100:.1f}%")
    print(f"avg any/completion     = {summary['avg_any_per_completion']:.2f}")
    print(f"\ntop error codes:")
    for code, cnt in summary["top_error_codes"]:
        print(f"  TS{code:5d}: {cnt} samples")
    print(f"\nsummary -> {out_path}")


if __name__ == "__main__":
    main()
