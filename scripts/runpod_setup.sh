#!/bin/bash
# RunPod セットアップスクリプト — yamato Phase 2 学習
#
# 推奨インスタンス:
#   RTX A5000 (24GB VRAM) ~$0.5-0.8/hr  → 学習 ~1-2h, 合計 ~$1-2
#   A100 SXM4 40GB        ~$1.5-2.0/hr  → 学習 ~45m-1h, 合計 ~$2-3
#
# 実行方法:
#   bash scripts/runpod_setup.sh
#   # セットアップ後:
#   bash scripts/runpod_train.sh

set -e

echo "=== yamatoLLM Phase 2 RunPod セットアップ ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ── 1. 依存パッケージ ────────────────────────────────────
echo ""
echo "[1/4] Python パッケージのインストール"
pip install -q \
    torch torchvision torchaudio \
    transformers==4.44.2 \
    peft==0.12.0 \
    bitsandbytes \
    accelerate \
    datasets \
    sentencepiece \
    protobuf \
    pyyaml \
    wandb \
    hydra-core \
    omegaconf

# llm-jp-eval 依存
pip install -q \
    "langchain>=0.1.0" \
    "langchain-community" \
    "openai" \
    "tiktoken" \
    "evaluate" \
    "rouge-score" \
    "sacrebleu"

# ── 2. リポジトリ ────────────────────────────────────────
echo ""
echo "[2/4] リポジトリのセットアップ"
if [ ! -d "yamato-public" ]; then
    git clone https://github.com/cosmopanda432/yamato-public.git
fi
cd yamato-public

# llm-jp-eval のセットアップ
if [ ! -d "external/llm-jp-eval" ]; then
    mkdir -p external
    git clone https://github.com/llm-jp/llm-jp-eval.git external/llm-jp-eval
    cd external/llm-jp-eval
    git checkout v1.4.1
    pip install -q -e .
    cd ../..
fi

# ── 3. モデルダウンロード ────────────────────────────────
echo ""
echo "[3/4] llm-jp-4-8b-base ダウンロード"
mkdir -p models
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='llm-jp/llm-jp-4-8b-base',
    local_dir='models/llm-jp-4-8b-base',
    ignore_patterns=['*.gguf'],
)
print('モデルダウンロード完了')
"

# ── 4. データセット前処理 ────────────────────────────────
echo ""
echo "[4/4] llm-jp-eval データセット前処理"
# JGLUE の URL を v1.3 に修正 (v1.1 は 404)
for f in jcommonsenseqa jnli jsquad; do
    sed -i 's/v1\.1/v1.3/g' \
        external/llm-jp-eval/src/llm_jp_eval/jaster/${f}.py 2>/dev/null || true
done

cd external/llm-jp-eval
python scripts/preprocess_dataset.py \
    --dataset-name all \
    --output-dir dataset/
cd ../..

echo ""
echo "=== セットアップ完了 ==="
echo "次のコマンドで学習を開始:"
echo "  bash scripts/runpod_train.sh"
