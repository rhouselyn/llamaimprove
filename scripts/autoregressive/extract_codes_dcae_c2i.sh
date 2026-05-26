#!/bin/bash
set -x

VQ_CKPT="/mnt/afs/zhengmingkai/whl/llamagen/tokenizer/tokenizer_image/results_dcae_training/033-DCAE-32-128b-256px/checkpoints/epoch_0040.pt"
DATA_PATH="/mnt/afs/zhengmingkai/whl/llamagen/imagenet_train_filelist.txt"
CODE_PATH="./dcae_codes"
BITS_PER_TOKEN=8
IMAGE_SIZE=256
NUM_GPUS=8

torchrun \
--nnodes=1 --nproc_per_node=${NUM_GPUS} --node_rank=0 \
--master_port=12336 \
autoregressive/train/extract_codes_dcae_c2i.py \
    --vq-ckpt ${VQ_CKPT} \
    --data-path ${DATA_PATH} \
    --code-path ${CODE_PATH} \
    --bits-per-token ${BITS_PER_TOKEN} \
    --dataset aoss \
    --aoss-bucket imagenet \
    --image-size ${IMAGE_SIZE} \
    --num-workers 16
