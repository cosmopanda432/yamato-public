"""
Phase 1: 国譲り — llm-jp-4-8b を yamatoLLM 骨格にロードする初期化スクリプト

ROADMAP Phase 1 Task 1:
    llm-jp-4-8b ロード → カスタムヘッド初期化 → テキスト生成確認

完了条件:
    RTX 3060 でテキスト生成が動く（量子化なし FP16/BF16 では VRAM 不足の可能性、
    --quantize 4bit を推奨）。

Usage:
    # 量子化なし（A100 等の大容量 GPU 向け）
    python scripts/kuniumi_init.py

    # 4bit 量子化（RTX 3060 向け）
    python scripts/kuniumi_init.py --quantize 4bit

    # プロンプト指定
    python scripts/kuniumi_init.py --quantize 4bit --prompt "日本の四季について教えてください。"
"""

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kojiki_lm.yamato_config import YamatoConfig
from kojiki_lm.yamato_model import YamatoLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: 国譲り — llm-jp-4-8b を yamatoLLM にロード",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="llm-jp/llm-jp-4-8b-base",
        help="HuggingFace モデル名（デフォルト: llm-jp/llm-jp-4-8b-base）",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["none", "4bit", "8bit"],
        default="none",
        help="量子化設定（RTX 3060 では 4bit 推奨）",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="こんにちは。あなたは誰ですか？",
        help="テスト用プロンプト",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="最大生成トークン数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--route",
        type=str,
        choices=["auto", "chat", "codegen", "retrieval"],
        default="auto",
        help="ルート強制指定（auto = 思兼神に判定させる）",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        default=None,
        help="国譲り後のチェックポイント保存先（オプション）",
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
        logging.info(
            "VRAM: %.1f GB (compute capability %d.%d)",
            props.total_memory / 1024**3,
            props.major,
            props.minor,
        )
        logging.info("BF16 supported: %s", torch.cuda.is_bf16_supported())


def kuniyuzuri(args: argparse.Namespace) -> YamatoLLM:
    """
    国譲り本体: llm-jp-4-8b をロードし、yamatoLLM 骨格に組み込む。

    YamatoLLM.from_qwen() が以下を実行する:
      1. llm-jp-4-8b のロード（量子化オプション付き）
      2. トークナイザーのロード
      3. カスタムヘッド（意図ルータ / 信頼度 / 言依さし）の初期化
      4. 浄化層（InbeSanitizer）の初期化
      5. stage を "kuniyuzuri" に設定
    """
    quantize = None if args.quantize == "none" else args.quantize

    logging.info("=" * 60)
    logging.info("国譲り (Kuniyuzuri) 開始")
    logging.info("  model: %s", args.model_name)
    logging.info("  quantize: %s", quantize)
    logging.info("=" * 60)

    config = YamatoConfig()
    config.backbone.model_name = args.model_name
    if quantize:
        config.inference.quantize = quantize

    t0 = time.perf_counter()
    model = YamatoLLM.from_qwen(
        model_name=args.model_name,
        quantize=quantize,
        config=config,
    )
    elapsed = time.perf_counter() - t0

    n_total = sum(p.numel() for p in model.parameters())
    n_custom = (
        sum(p.numel() for p in model.custom_heads.parameters())
        if model.custom_heads is not None
        else 0
    )
    logging.info(
        "国譲り完了 (%.1fs): backbone+heads = %.2fB params (custom heads = %.2fM)",
        elapsed,
        n_total / 1e9,
        n_custom / 1e6,
    )

    return model


def test_generation(model: YamatoLLM, args: argparse.Namespace) -> None:
    logging.info("=" * 60)
    logging.info("生成テスト")
    logging.info("  prompt: %s", args.prompt)
    logging.info("=" * 60)

    route_arg = None if args.route == "auto" else args.route

    t0 = time.perf_counter()
    output = model.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        route=route_arg,
    )
    elapsed = time.perf_counter() - t0

    logging.info("生成完了 (%.2fs)", elapsed)
    logging.info("  route: %s", output.route)
    logging.info("  confidence: %.3f", output.confidence)
    logging.info("  uncertainty_flag: %s", output.uncertainty_flag)
    logging.info("  truthfulness: %.3f", output.truthfulness)
    logging.info("  safety_score: %.3f", output.safety_score)
    if output.verdict:
        logging.info("  verdict: %s", output.verdict)

    print("\n----- 生成テキスト -----")
    print(output.text)
    print("------------------------\n")


def maybe_save_checkpoint(model: YamatoLLM, args: argparse.Namespace) -> None:
    if not args.save_checkpoint:
        return

    import torch

    out_path = Path(args.save_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "config": model.config,
        "base_model_name": model.config.backbone.model_name,
        "stage": model.config.stage,
        "model_state_dict": {
            k: v.cpu() for k, v in model.custom_heads.state_dict().items()
        }
        if model.custom_heads is not None
        else {},
    }
    torch.save(checkpoint, out_path)
    logging.info("チェックポイント保存: %s", out_path)


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        report_environment()
        model = kuniyuzuri(args)
        test_generation(model, args)
        maybe_save_checkpoint(model, args)
    except Exception as exc:
        logging.exception("国譲り失敗: %s", exc)
        return 1

    logging.info("Phase 1 Task 1 完了 — 国譲り成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
