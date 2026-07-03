#!/bin/bash

# 单GPU训练脚本
export CUDA_VISIBLE_DEVICES=0  # 只使用GPU 0

echo "Starting single GPU training..."
echo "Using GPU 0 only"
echo "Data directory: /root/autodl-tmp/optimum-main/examples/cifar-10-batches-py"

python train_mor_vit_cifar10.py \
    --data_dir /root/autodl-tmp/optimum-main/examples/cifar-10-batches-py \
    --batch_size 256 \
    --epochs 100 \
    --lr 6e-5 \
    --use_pretrained \
    --enable_dora

echo "Training completed!"
