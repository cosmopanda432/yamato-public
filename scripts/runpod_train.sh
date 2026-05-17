#!/bin/bash
# RunPod 学習 — yamatoLLM Qwen2.5-Coder-7B + LoRA + TsukuyomiTypeHead
#
# runpod_setup.sh 完了後に実行。
# GPU VRAM を見て LoRA rank / batch / seq_len を自動調整する。

set -e
cd "$(dirname "$0")/.."

VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{print int($1/1024)}')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
echo "=== yamatoLLM SFT (Qwen2.5-Coder-7B) ==="
echo "GPU: ${GPU_NAME} (${VRAM_GB}GB VRAM)"

if [ "$VRAM_GB" -ge 70 ]; then
    # A100 80GB / H100 80GB — 余裕があるので grad_checkpointing を切って高速化
    LORA_R=32
    BATCH=2
    GRAD_ACCUM=8
    MAX_SEQ=1024
    QUANTIZE="4bit"
    EXTRA_FLAGS="--no-grad-checkpoint"
    MAX_STEPS=1500
    PROFILE="A100/H100 80GB (QLoRA r=32, batch=2, seq=1024, no-grad-ckpt, max-steps=1500)"
elif [ "$VRAM_GB" -ge 35 ]; then
    # A100 40GB / L40S
    LORA_R=32
    BATCH=1
    GRAD_ACCUM=16
    MAX_SEQ=2048
    QUANTIZE="4bit"
    EXTRA_FLAGS=""
    MAX_STEPS=-1
    PROFILE="A100-40GB / L40S (QLoRA, r=32, batch=1, seq=2048)"
elif [ "$VRAM_GB" -ge 20 ]; then
    # RTX 4090 / A5000 / 3090
    LORA_R=16
    BATCH=1
    GRAD_ACCUM=16
    MAX_SEQ=1536
    QUANTIZE="4bit"
    EXTRA_FLAGS=""
    MAX_STEPS=-1
    PROFILE="4090/A5000/3090 (QLoRA r=16, batch=1, seq=1536)"
else
    # RTX 3060 12GB 等
    LORA_R=8
    BATCH=1
    GRAD_ACCUM=8
    MAX_SEQ=512
    QUANTIZE="4bit"
    EXTRA_FLAGS=""
    MAX_STEPS=-1
    PROFILE="<20GB (QLoRA r=8, batch=1, seq=512)"
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
    --lora-r "${LORA_R}" \
    --lora-alpha "$((LORA_R * 2))" \
    --batch-size "${BATCH}" \
    --grad-accum "${GRAD_ACCUM}" \
    --max-seq-length "${MAX_SEQ}" \
    --num-epochs 1 \
    --max-steps "${MAX_STEPS}" \
    --learning-rate 2e-4 \
    --head-lr 1e-3 \
    --type-loss-weight 0.3 \
    ${EXTRA_FLAGS} \
    --log-every 25 \
    --save-every 500 \
    2>&1

echo ""
echo "=== 学習完了 ==="
echo "checkpoints/yamato_sft/final/ に LoRA adapter + custom_heads.pt が保存されました"
echo ""
echo "評価する場合:"
echo "  python3 scripts/eval/generate_multipl_e.py \\"
echo "      --input data/raw/multipl_e/humaneval-ts/test-00000-of-00001.parquet \\"
echo "      --out-dir data/eval/generated/humaneval-ts-yamato \\"
echo "      --model checkpoints/yamato_sft/final/lora_adapter"
echo "  # ※ 評価には LoRA を merge するか、from_pretrained で読む側の対応が必要"
