#!/bin/bash
# RunPod 学習スクリプト — yamato Phase 2
#
# runpod_setup.sh 完了後に実行する。
# GPU VRAM に応じて自動でパラメータを調整する。

set -e
cd "$(dirname "$0")/.."

# GPU VRAM 検出
VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print int($1/1024)}')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "=== yamato Phase 2 学習 ==="
echo "GPU: ${GPU_NAME} (${VRAM_GB}GB VRAM)"

# VRAM ごとのパラメータ設定
if [ "$VRAM_GB" -ge 40 ]; then
    # A100 40GB / A6000 48GB: 余裕あり
    LORA_R=32
    MAX_SAMPLES=1000
    BATCH=2
    GRAD_ACCUM=8
    MAX_SEQ=1024
    echo "設定: A100/A6000 高品質モード (LoRA r=32, samples=3000)"
elif [ "$VRAM_GB" -ge 20 ]; then
    # RTX A5000 24GB / RTX 3090 24GB
    LORA_R=16
    MAX_SAMPLES=500
    BATCH=1
    GRAD_ACCUM=16
    MAX_SEQ=512
    echo "設定: A5000/3090 標準モード (LoRA r=16, samples=1500)"
else
    # RTX 3060 12GB 等 (ローカル動作確認用)
    LORA_R=8
    MAX_SAMPLES=200
    BATCH=1
    GRAD_ACCUM=16
    MAX_SEQ=512
    echo "設定: 12GB 省メモリモード (LoRA r=8, samples=600)"
fi

# 学習実行
PYTORCH_ALLOC_CONF=expandable_segments:True python3 scripts/train_phase2.py \
    --model-name models/llm-jp-4-8b-base \
    --output-dir checkpoints/phase2 \
    --lora-r "${LORA_R}" \
    --lora-alpha "$((LORA_R * 2))" \
    --max-samples "${MAX_SAMPLES}" \
    --batch-size "${BATCH}" \
    --grad-accum "${GRAD_ACCUM}" \
    --max-seq-length "${MAX_SEQ}" \
    --num-epochs 3 \
    --learning-rate 2e-4 \
    --head-lr 1e-3

echo ""
echo "=== 学習完了 → LoRA マージ ==="
python3 scripts/merge_lora.py \
    --checkpoint-dir checkpoints/phase2 \
    --base-model models/llm-jp-4-8b-base

echo ""
echo "=== llm-jp-eval 評価 (yamatoLLM) ==="
python3 scripts/eval_baseline.py \
    --model-name checkpoints/phase2/merged \
    --llm-jp-eval-path external/llm-jp-eval \
    --output-dir results/phase2 \
    --max-num-samples 100

echo ""
echo "=== ベースライン比較 ==="
echo "Baseline (Phase 1):  JCommonsenseQA=0.91 / JNLI=0.92 / JSQuAD=0.896"
echo "yamatoLLM (Phase 2): 上記 results/phase2/ を参照"
