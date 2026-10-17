#!/bin/bash

GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4

torchrun \
    --nproc_per_node="${GPU_NUM}" \
    train.py \
    --output_dir "${OUTPUT_DIR}" \
    -c "${CFG}" \
    --datasets "${DATASETS}" \
    --log_tensorboard \
    --amp
