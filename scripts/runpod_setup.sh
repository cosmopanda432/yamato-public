#!/bin/bash
# RunPod セットアップ — yamatoLLM Qwen2.5-Coder-7B SFT
#
# 前提:
#   1) git clone してリポ内に入っていること
#        git clone https://github.com/cosmopanda432/yamato-public.git
#        cd yamato-public
#        git checkout master      # 重要: デフォルトは main (Initial commit のみ)
#   2) このスクリプトを `bash scripts/runpod_setup.sh` で実行する
#      → 内部で repo root に cd するので、どこから呼んでも安全
#
# 推奨インスタンス:
#   RTX A5000 24GB        ~$0.4-0.6/hr  → 学習 1-3h, 合計 ~$2-5
#   RTX 4090 24GB         ~$0.4-0.7/hr  → 学習 1-2h, 合計 ~$2-4
#   A100 SXM4 80GB        ~$1.5-2.0/hr  → 学習 30m-1h, 合計 ~$2-3
#   H100 SXM5 80GB        ~$2.5-4.0/hr  → 学習 20-40m, 合計 ~$3-5

set -e

# repo root に cd (スクリプトの所在から逆算 → 二重 dir 事故を防ぐ)
cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
echo "=== yamatoLLM RunPod セットアップ ==="
echo "Repo:   ${REPO_ROOT}"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)  ($(git rev-parse --short HEAD))"
echo "GPU:    $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ── 1. 依存パッケージ ────────────────────────────────────
echo ""
echo "[1/4] Python パッケージのインストール"
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

# ── 2. モデルダウンロード ────────────────────────────────
echo ""
echo "[2/4] Qwen2.5-Coder-7B-Instruct ダウンロード (~15GB)"
mkdir -p models
if [ ! -f "models/Qwen2.5-Coder-7B-Instruct/config.json" ]; then
    huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
        --local-dir models/Qwen2.5-Coder-7B-Instruct \
        --exclude "*.gguf" "original/*"
else
    echo "  既存検出: models/Qwen2.5-Coder-7B-Instruct/ (スキップ)"
fi

# ── 3. データ取得 ────────────────────────────────────────
echo ""
echo "[3/4] ManyTypes4TypeScript + MultiPL-E TS をダウンロード"
mkdir -p data/raw
if [ ! -d "data/raw/many_types_4_ts/data" ]; then
    huggingface-cli download kevinjesse/ManyTypes4TypeScript --repo-type dataset \
        --local-dir data/raw/many_types_4_ts
else
    echo "  既存検出: data/raw/many_types_4_ts/ (スキップ)"
fi
if [ ! -d "data/raw/multipl_e/humaneval-ts" ]; then
    huggingface-cli download nuprl/MultiPL-E --repo-type dataset \
        --include "humaneval-ts/*" "mbpp-ts/*" \
        --local-dir data/raw/multipl_e
else
    echo "  既存検出: data/raw/multipl_e/ (スキップ)"
fi

# ── 4. SFT 用 parquet を生成 (随時 flush、進捗ログ付き) ───
echo ""
echo "[4/4] SFT 用データ前処理 (ストリーミング書き出し)"
mkdir -p data/processed/sft

if [ ! -f "data/processed/sft/train.parquet" ]; then
    echo "  -> train.parquet 生成中 (進捗は 2000 行ごとに表示)"
    python3 -u scripts/data/prepare_sft_dataset.py \
        --split train \
        --out data/processed/sft/train.parquet \
        --max-seq-len 2048
else
    echo "  既存検出: data/processed/sft/train.parquet (スキップ)"
fi

if [ ! -f "data/processed/sft/validation.parquet" ]; then
    echo "  -> validation.parquet 生成中"
    python3 -u scripts/data/prepare_sft_dataset.py \
        --split validation \
        --out data/processed/sft/validation.parquet \
        --max-seq-len 2048
else
    echo "  既存検出: data/processed/sft/validation.parquet (スキップ)"
fi

echo ""
echo "=== セットアップ完了 ==="
echo "次のコマンドで学習開始:"
echo "  bash scripts/runpod_train.sh"
