#!/bin/bash
set -e
set -x

VQ_CKPT="/mnt/afs/zhengmingkai/whl/llamagen/tokenizer/tokenizer_image/results_dcae_training/033-DCAE-32-128b-256px/checkpoints/epoch_0040.pt"
DATA_PATH="/mnt/afs/zhengmingkai/whl/llamagen/imagenet_train_filelist.txt"
CODE_PATH="./dcae_codes"
CLOUD_SAVE_PATH="./results_dcae_ar_direct"
SAMPLE_DIR="./samples_dcae"
GPT_MODEL="GPT-B"
CODEBOOK_DIM=128
IMAGE_SIZE=256
DOWNSAMPLE_SIZE=32
NUM_CLASSES=1000
EPOCHS=300
LR=1e-4
GLOBAL_BATCH_SIZE=256
NUM_GPUS=8
CFG_SCALE=1.5
TEMPERATURE=1.0
NUM_FID_SAMPLES=50000
EVAL_EVERY=10
CKPT_EVERY=5000

echo "============================================================"
echo "  DCAE + LlamaGen Autoregressive Training Pipeline"
echo "  Approach: 128-dim continuous output + MSE + sign()"
echo "============================================================"
echo ""
echo "  VQ_CKPT:           ${VQ_CKPT}"
echo "  DATA_PATH:         ${DATA_PATH}"
echo "  CODE_PATH:         ${CODE_PATH}"
echo "  GPT_MODEL:         ${GPT_MODEL}"
echo "  CODEBOOK_DIM:      ${CODEBOOK_DIM}"
echo "  IMAGE_SIZE:        ${IMAGE_SIZE}"
echo "  NUM_GPUS:          ${NUM_GPUS}"
echo "  EPOCHS:            ${EPOCHS}"
echo "  EVAL_EVERY:        ${EVAL_EVERY} epochs"
echo "  CKPT_EVERY:        ${CKPT_EVERY} steps"
echo "  CFG_SCALE:         ${CFG_SCALE}"
echo "============================================================"
echo ""

# ─── Step 1: Extract codes ────────────────────────────────────
echo ""
echo "========== Step 1/3: Extracting binary codes from DCAE =========="
echo ""

CODE_DIR="${CODE_PATH}/imagenet${IMAGE_SIZE}_codes_dcae_direct"
if [ -d "$CODE_DIR" ] && [ "$(ls -A $CODE_DIR 2>/dev/null)" ]; then
    echo "Codes already exist at ${CODE_DIR}, skipping extraction."
    echo "To re-extract, delete the directory first: rm -rf ${CODE_DIR}"
else
    torchrun \
    --nnodes=1 --nproc_per_node=${NUM_GPUS} --node_rank=0 \
    --master_port=12338 \
    autoregressive/train/extract_codes_dcae_direct.py \
        --vq-ckpt ${VQ_CKPT} \
        --data-path ${DATA_PATH} \
        --code-path ${CODE_PATH} \
        --dataset aoss \
        --aoss-bucket imagenet \
        --image-size ${IMAGE_SIZE} \
        --num-workers 16

    echo "Code extraction complete!"
fi

echo ""
echo "Codes saved at: ${CODE_DIR}"
echo "Label dir: ${CODE_PATH}/imagenet${IMAGE_SIZE}_labels"
echo ""

# ─── Step 2: Train autoregressive model ───────────────────────
echo ""
echo "========== Step 2/3: Training autoregressive model =========="
echo ""

torchrun \
--nnodes=1 --nproc_per_node=${NUM_GPUS} --node_rank=0 \
--master_port=12339 \
autoregressive/train/train_dcae_c2i_direct.py \
    --code-path ${CODE_PATH} \
    --cloud-save-path ${CLOUD_SAVE_PATH} \
    --gpt-model ${GPT_MODEL} \
    --gpt-type c2i \
    --codebook-dim ${CODEBOOK_DIM} \
    --image-size ${IMAGE_SIZE} \
    --downsample-size ${DOWNSAMPLE_SIZE} \
    --num-classes ${NUM_CLASSES} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --mixed-precision bf16 \
    --ema \
    --eval-every ${EVAL_EVERY} \
    --ckpt-every ${CKPT_EVERY} \
    --log-every 100 \
    --num-workers 16

echo ""
echo "Training complete!"
echo ""

# ─── Step 3: Evaluate (sample + FID/IS) ───────────────────────
echo ""
echo "========== Step 3/3: Evaluating model =========="
echo ""

LATEST_CKPT=$(find ${CLOUD_SAVE_PATH} -name "*.pt" -path "*/checkpoints/*" | sort | tail -1)
if [ -z "$LATEST_CKPT" ]; then
    LATEST_CKPT=$(find results -name "*.pt" -path "*/checkpoints/*" | sort | tail -1)
fi

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found! Cannot evaluate."
    exit 1
fi

echo "Using checkpoint: ${LATEST_CKPT}"
echo ""

python autoregressive/eval/eval_dcae_c2i.py \
    --gpt-ckpt ${LATEST_CKPT} \
    --vq-ckpt ${VQ_CKPT} \
    --gpt-model ${GPT_MODEL} \
    --image-size ${IMAGE_SIZE} \
    --num-classes ${NUM_CLASSES} \
    --cfg-scale ${CFG_SCALE} \
    --temperature ${TEMPERATURE} \
    --num-fid-samples ${NUM_FID_SAMPLES} \
    --num-gpus ${NUM_GPUS} \
    --sample-dir ${SAMPLE_DIR}

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Codes:     ${CODE_DIR}"
echo "  Checkpoints: ${CLOUD_SAVE_PATH}"
echo "  Samples:   ${SAMPLE_DIR}"
echo "============================================================"
