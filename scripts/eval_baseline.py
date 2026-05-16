"""
Phase 1 Task 3: ベースライン計測 — llm-jp-4-8b (素) を llm-jp-eval で評価

ROADMAP Phase 1 Task 3:
    llm-jp-4-8b (素) を llm-jp-eval で評価 → 比較基準を取得

llm-jp-eval は外部ツールのためサブプロセスで呼び出す方式とする。
このスクリプトは llm-jp-eval のラッパー:
    - 設定ファイルの生成
    - サブプロセス実行
    - 結果サマリの抽出

事前準備:
    git clone https://github.com/llm-jp/llm-jp-eval.git external/llm-jp-eval
    cd external/llm-jp-eval
    pip install -e .
    python scripts/preprocess_dataset.py --dataset-name all --output-dir dataset/

Usage:
    # ベースライン評価（FP16）
    python scripts/eval_baseline.py --llm-jp-eval-path external/llm-jp-eval

    # 量子化版の評価
    python scripts/eval_baseline.py --llm-jp-eval-path external/llm-jp-eval --quantize 4bit

    # 結果のサマリ表示のみ（再評価せず）
    python scripts/eval_baseline.py --summary-only --output-dir results/baseline/
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 Task 3: llm-jp-4-8b ベースラインを llm-jp-eval で評価",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="llm-jp/llm-jp-4-8b-base",
    )
    parser.add_argument(
        "--llm-jp-eval-path",
        type=str,
        default=str(REPO_ROOT / "external" / "llm-jp-eval"),
        help="llm-jp-eval リポジトリのパス",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "baseline"),
        help="評価結果出力ディレクトリ",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "4bit", "8bit"],
        default="none",
    )
    parser.add_argument(
        "--target-datasets",
        type=str,
        nargs="+",
        default=["jcommonsenseqa", "jnli", "jsquad", "xlsum_ja"],
        help="評価対象データセット（ROADMAP Phase 5 と合わせる）",
    )
    parser.add_argument(
        "--max-num-samples",
        type=int,
        default=100,
        help="各データセットの評価サンプル数上限",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="評価実行せず既存結果のサマリのみ表示",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="設定生成までで止めて実行しない",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def check_prerequisites(eval_path: Path) -> None:
    if not eval_path.exists():
        raise FileNotFoundError(
            f"llm-jp-eval が見つかりません: {eval_path}\n"
            "事前準備:\n"
            f"  git clone https://github.com/llm-jp/llm-jp-eval.git {eval_path}\n"
            f"  cd {eval_path} && pip install -e .\n"
            "  python scripts/preprocess_dataset.py --dataset-name all --output-dir dataset/"
        )

    dataset_dir = eval_path / "dataset"
    if not dataset_dir.exists():
        logging.warning(
            "データセットが未準備の可能性: %s が存在しません。"
            "llm-jp-eval の preprocess_dataset.py を先に実行してください。",
            dataset_dir,
        )


def build_eval_config(args: argparse.Namespace, output_dir: Path) -> Path:
    """llm-jp-eval の config ファイルを生成"""
    quantize = None if args.quantize == "none" else args.quantize

    config = {
        "model": {
            "pretrained_model_name_or_path": args.model_name,
            "trust_remote_code": True,
            "device_map": "auto",
        },
        "tokenizer": {
            "pretrained_model_name_or_path": args.model_name,
            "trust_remote_code": True,
        },
        "target_dataset": args.target_datasets,
        "log_dir": str(output_dir / "logs"),
        "metainfo": {
            "version": "phase1-baseline",
            "model_name": args.model_name,
            "quantize": quantize or "fp16",
            "basemodel_name": args.model_name,
            "model_type": "llama",
            "instruction_tuning_method_by_llm_jp": "None",
            "instruction_tuning_data_by_llm_jp": [],
        },
        "max_num_samples": args.max_num_samples,
        "torch_dtype": "bf16",
        "wandb": {
            "log": False,
        },
    }

    if quantize == "4bit":
        config["model"]["load_in_4bit"] = True
    elif quantize == "8bit":
        config["model"]["load_in_8bit"] = True

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "eval_config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logging.info("評価設定を生成: %s", config_path)
    return config_path


def run_eval(eval_path: Path, config_path: Path) -> int:
    """llm-jp-eval をサブプロセスで実行"""
    eval_script = eval_path / "scripts" / "evaluate_llm.py"
    if not eval_script.exists():
        # 新しい llm-jp-eval だとパスが異なる可能性 — フォールバック
        candidates = list(eval_path.rglob("evaluate_llm.py"))
        if candidates:
            eval_script = candidates[0]
        else:
            raise FileNotFoundError(
                f"evaluate_llm.py が見つかりません ({eval_path} 配下を探索)"
            )

    cmd = [
        sys.executable,
        str(eval_script),
        "-cn",
        str(config_path),
    ]
    logging.info("実行: %s", " ".join(cmd))

    result = subprocess.run(cmd, cwd=str(eval_path))
    return result.returncode


def summarize_results(output_dir: Path) -> None:
    log_dir = output_dir / "logs"
    if not log_dir.exists():
        logging.warning("結果ディレクトリが見つかりません: %s", log_dir)
        return

    score_files = list(log_dir.rglob("*.json"))
    if not score_files:
        logging.warning("スコアファイルが見つかりません: %s", log_dir)
        return

    logging.info("=" * 60)
    logging.info("ベースラインスコア サマリ")
    logging.info("=" * 60)

    for sf in sorted(score_files):
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logging.warning("読み込み失敗 %s: %s", sf, exc)
            continue

        scores = data.get("scores") or data
        if isinstance(scores, dict):
            for dataset, score in scores.items():
                if isinstance(score, (int, float)):
                    logging.info("  %s: %.4f", dataset, score)
                elif isinstance(score, dict):
                    for metric, value in score.items():
                        if isinstance(value, (int, float)):
                            logging.info("  %s/%s: %.4f", dataset, metric, value)


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.summary_only:
            summarize_results(output_dir)
            return 0

        eval_path = Path(args.llm_jp_eval_path)
        check_prerequisites(eval_path)

        config_path = build_eval_config(args, output_dir)

        if args.dry_run:
            logging.info("--dry-run: 評価は実行しません")
            return 0

        rc = run_eval(eval_path, config_path)
        if rc != 0:
            logging.error("llm-jp-eval 実行失敗 (rc=%d)", rc)
            return rc

        summarize_results(output_dir)
    except Exception as exc:
        logging.exception("ベースライン評価失敗: %s", exc)
        return 1

    logging.info("Phase 1 Task 3 完了 — ベースライン取得")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
