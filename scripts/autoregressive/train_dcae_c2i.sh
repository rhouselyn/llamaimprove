#!/bin/bash
set -x

CODE_PATH="./dcae_codes"
CLOUD_SAVE_PATH="./results_dcae_ar"
GPT_MODEL="GPT-B"
VOCAB_SIZE=256
NUM_BITS=128
BITS_PER_TOKEN=8
IMAGE_SIZE=256
DOWNSAMPLE_SIZE=32
NUM_CLASSES=1000
EPOCHS=300
LR=1e-4
GLOBAL_BATCH_SIZE=256
NUM_GPUS=8

torchrun \
--nnodes=1 --nproc_per_node=${NUM_GPUS} --node_rank=0 \
--master_port=12337 \
autoregressive/train/train_dcae_c2i.py \
    --code-path ${CODE_PATH} \
    --cloud-save-path ${CLOUD_SAVE_PATH} \
    --gpt-model ${GPT_MODEL} \
    --gpt-type c2i \
    --vocab-size ${VOCAB_SIZE} \
    --num-bits ${NUM_BITS} \
    --bits-per-token ${BITS_PER_TOKEN} \
    --image-size ${IMAGE_SIZE} \
    --downsample-size ${DOWNSAMPLE_SIZE} \
    --num-classes ${NUM_CLASSES} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --mixed-precision bf16 \
    --ema \
    --num-workers 16
