"""
MultiPL-E TS の生成結果に対してテストを実行し pass@1 を集計する。

各サンプルごとに:
    1. 一時 .ts ファイルに `prompt + completion + tests` を書き出す
    2. `npx tsx <file>` で実行
    3. exit code 0 = pass, それ以外 = fail
    4. stderr 先頭 5 行をエラーメッセージとして記録

出力:
    data/eval/results/<dataset>/<name>__s<i>.result.json
    data/eval/results/<dataset>/_summary.json   (pass@1, n_total, ...)

使い方:
    python3 scripts/eval/run_tests.py \
        --generated-dir data/eval/generated/humaneval-ts \
        --out-dir data/eval/results/humaneval-ts
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

TSX_BIN = "scripts/ts_tools/node_modules/.bin/tsx"


def run_one(prompt: str, completion: str, tests: str, timeout: float) -> dict:
    """1サンプルを評価して結果を返す"""
    source = prompt + completion + "\n\n" + tests + "\n"
    with tempfile.NamedTemporaryFile(
        "w", suffix=".ts", delete=False, dir="/tmp"
    ) as f:
        f.write(source)
        tmp_path = f.name

    t0 = time.time()
    try:
        proc = subprocess.run(
            [TSX_BIN, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        stderr_lines = proc.stderr.strip().split("\n")[:5]
        stdout_tail = proc.stdout.strip().split("\n")[-3:]
        return {
            "ok": ok,
            "exit_code": proc.returncode,
            "elapsed_sec": elapsed,
            "stderr_head": stderr_lines,
            "stdout_tail": stdout_tail,
            "timeout": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "exit_code": None,
            "elapsed_sec": time.time() - t0,
            "stderr_head": ["TIMEOUT"],
            "stdout_tail": [],
            "timeout": True,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    gen_dir = Path(args.generated_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(TSX_BIN).exists():
        raise SystemExit(
            f"{TSX_BIN} not found. Run `cd scripts/ts_tools && npm install` first."
        )

    files = sorted(gen_dir.glob("*.json"))
    if args.limit is not None:
        files = files[: args.limit]
    print(f"Evaluating {len(files)} samples from {gen_dir}")

    results = []
    n_pass = 0
    t_total = 0.0
    for i, jf in enumerate(files):
        data = json.loads(jf.read_text())
        res = run_one(
            prompt=data["prompt"],
            completion=data["completion"],
            tests=data["tests"],
            timeout=args.timeout,
        )
        t_total += res["elapsed_sec"]
        if res["ok"]:
            n_pass += 1

        out_path = out_dir / jf.name.replace(".json", ".result.json")
        out_path.write_text(json.dumps({
            "name": data["name"],
            "sample_id": data["sample_id"],
            **res,
        }, ensure_ascii=False))

        results.append({
            "name": data["name"],
            "ok": res["ok"],
            "exit_code": res["exit_code"],
            "timeout": res["timeout"],
        })

        status = "PASS" if res["ok"] else ("TIMEOUT" if res["timeout"] else "FAIL")
        print(f"  [{i+1}/{len(files)}] {data['name']:50s} {status:7s} ({res['elapsed_sec']:.1f}s)")

    n_total = len(files)
    pass_at_1 = n_pass / max(n_total, 1)
    summary = {
        "n_total": n_total,
        "n_pass": n_pass,
        "pass_at_1": pass_at_1,
        "avg_runtime_sec": t_total / max(n_total, 1),
        "details": results,
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\npass@1 = {n_pass}/{n_total} = {pass_at_1 * 100:.1f}%")
    print(f"avg runtime = {t_total / max(n_total, 1):.1f}s/test, total {t_total:.0f}s")
    print(f"summary -> {out_dir / '_summary.json'}")


if __name__ == "__main__":
    main()
