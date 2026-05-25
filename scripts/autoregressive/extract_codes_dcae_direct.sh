#!/bin/bash
set -x

VQ_CKPT="/mnt/afs/zhengmingkai/whl/llamagen/tokenizer/tokenizer_image/results_dcae_training/033-DCAE-32-128b-256px/checkpoints/epoch_0040.pt"
DATA_PATH="/mnt/afs/zhengmingkai/whl/llamagen/imagenet_train_filelist.txt"
CODE_PATH="./dcae_codes"
IMAGE_SIZE=256
NUM_GPUS=8

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
