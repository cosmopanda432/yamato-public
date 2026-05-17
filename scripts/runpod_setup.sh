#!/bin/bash
# RunPod セットアップ — yamatoLLM Qwen2.5-Coder-7B SFT
#
# 推奨インスタンス:
#   RTX A5000 24GB        ~$0.4-0.6/hr  → 学習 1-3h, 合計 ~$2-5
#   RTX 4090 24GB         ~$0.4-0.7/hr  → 学習 1-2h, 合計 ~$2-4
#   A100 SXM4 80GB        ~$1.5-2.0/hr  → 学習 30m-1h, 合計 ~$2-3
#   H100 SXM5 80GB        ~$2.5-4.0/hr  → 学習 20-40m, 合計 ~$3-5
#
# 実行手順:
#   bash scripts/runpod_setup.sh
#   bash scripts/runpod_train.sh

set -e

echo "=== yamatoLLM RunPod セットアップ ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ── 1. 依存パッケージ ────────────────────────────────────
echo ""
echo "[1/5] Python パッケージのインストール"
pip install -q \
    "torch>=2.0" \
    "transformers>=4.44,<4.60" \
    "peft>=0.12" \
    "bitsandbytes>=0.43" \
    "accelerate>=0.30" \
    "datasets>=2.18" \
    "pyarrow>=15" \
    "huggingface_hub" \
    "sentencepiece" \
    "wandb"

# ── 2. リポジトリ ────────────────────────────────────────
echo ""
echo "[2/5] リポジトリのセットアップ"
if [ ! -d "yamato-public" ]; then
    git clone https://github.com/cosmopanda432/yamato-public.git
fi
cd yamato-public
git pull --ff-only

# ── 3. モデルダウンロード ────────────────────────────────
echo ""
echo "[3/5] Qwen2.5-Coder-7B-Instruct ダウンロード"
mkdir -p models
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
    --local-dir models/Qwen2.5-Coder-7B-Instruct \
    --exclude "*.gguf" "original/*"

# ── 4. データ取得 ────────────────────────────────────────
echo ""
echo "[4/5] ManyTypes4TypeScript + MultiPL-E TS をダウンロード"
mkdir -p data/raw
if [ ! -d "data/raw/many_types_4_ts/data" ]; then
    huggingface-cli download kevinjesse/ManyTypes4TypeScript --repo-type dataset \
        --local-dir data/raw/many_types_4_ts
fi
if [ ! -d "data/raw/multipl_e/humaneval-ts" ]; then
    huggingface-cli download nuprl/MultiPL-E --repo-type dataset \
        --include "humaneval-ts/*" "mbpp-ts/*" \
        --local-dir data/raw/multipl_e
fi

# ── 5. SFT 用 parquet を生成 ─────────────────────────────
echo ""
echo "[5/5] SFT 用データ前処理"
mkdir -p data/processed/sft

# train split: 全量 (RunPod 上で十分速い)
if [ ! -f "data/processed/sft/train.parquet" ]; then
    python3 scripts/data/prepare_sft_dataset.py \
        --split train \
        --out data/processed/sft/train.parquet \
        --max-seq-len 2048
fi

# validation split
if [ ! -f "data/processed/sft/validation.parquet" ]; then
    python3 scripts/data/prepare_sft_dataset.py \
        --split validation \
        --out data/processed/sft/validation.parquet \
        --max-seq-len 2048
fi

echo ""
echo "=== セットアップ完了 ==="
echo "次のコマンドで学習開始:"
echo "  bash scripts/runpod_train.sh"
