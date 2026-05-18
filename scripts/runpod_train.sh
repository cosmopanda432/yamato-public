#!/bin/bash
# RunPod 学習 — yamatoLLM Qwen2.5-Coder-7B (frozen) + TsukuyomiTypeHead + BonpuConfidence
#
# backbone は完全 freeze、heads (14.82M) のみ学習する。
# runpod_setup.sh 完了後に実行。

set -e
cd "$(dirname "$0")/.."

VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print int($1/1024)}')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "=== yamatoLLM SFT (Qwen2.5-Coder-7B, backbone FROZEN) ==="
echo "GPU: ${GPU_NAME} (${VRAM_GB}GB VRAM)"

# backbone は 4bit quantize で固定。heads は 14.82M。
# VRAM 主要因は (batch * seq * hidden) と head の grad/optim state のみ。
if [ "$VRAM_GB" -ge 70 ]; then
    # A100 80GB / H100 80GB
    BATCH=8
    GRAD_ACCUM=2
    MAX_SEQ=2048
    QUANTIZE="4bit"
    MAX_STEPS=-1
    PROFILE="A100/H100 80GB (batch=8, seq=2048, all data 1 epoch)"
elif [ "$VRAM_GB" -ge 35 ]; then
    # A100 40GB / L40S
    BATCH=4
    GRAD_ACCUM=4
    MAX_SEQ=2048
    QUANTIZE="4bit"
    MAX_STEPS=-1
    PROFILE="A100-40GB / L40S (batch=4, seq=2048)"
elif [ "$VRAM_GB" -ge 20 ]; then
    # RTX 4090 / A5000 / 3090
    BATCH=2
    GRAD_ACCUM=8
    MAX_SEQ=1536
    QUANTIZE="4bit"
    MAX_STEPS=-1
    PROFILE="4090/A5000/3090 (batch=2, seq=1536)"
else
    # RTX 3060 12GB 等
    BATCH=1
    GRAD_ACCUM=8
    MAX_SEQ=1024
    QUANTIZE="4bit"
    MAX_STEPS=-1
    PROFILE="<20GB (batch=1, seq=1024)"
fi

echo "Profile: ${PROFILE}"

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTORCH_ALLOC_CONF=expandable_segments:True \
python3 -u scripts/train/sft_yamato.py \
    --train-parquet data/processed/sft/train.parquet \
    --val-parquet data/processed/sft/validation.parquet \
    --output-dir checkpoints/yamato_sft \
    --quantize "${QUANTIZE}" \
    --batch-size "${BATCH}" \
    --grad-accum "${GRAD_ACCUM}" \
    --max-seq-length "${MAX_SEQ}" \
    --num-epochs 1 \
    --max-steps "${MAX_STEPS}" \
    --head-lr 1e-3 \
    --type-loss-weight 1.0 \
    --conf-loss-weight 0.3 \
    --log-every 25 \
    --save-every 500 \
    2>&1

echo ""
echo "=== 学習完了 ==="
echo "checkpoints/yamato_sft/final/custom_heads.pt が保存されました"
echo "(backbone は不変なので保存不要、推論時に base Qwen + heads を attach)"
