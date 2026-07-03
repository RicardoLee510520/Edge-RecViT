#!/bin/bash

# 设置NCCL环境变量以避免超时和提高性能
export NCCL_DEBUG=INFO  # 启用详细日志
export NCCL_TIMEOUT=1800  # 设置超时时间为30分钟（默认是10分钟）
export NCCL_IB_DISABLE=1  # 如果没有InfiniBand，禁用它
export NCCL_P2P_DISABLE=0  # 启用P2P通信
export CUDA_VISIBLE_DEVICES=0,1  # 使用GPU 0和1

# 设置PyTorch环境变量
export TORCH_DISTRIBUTED_DEBUG=INFO  # 启用分布式调试信息
export OMP_NUM_THREADS=4  # 限制每个进程的线程数

# 运行分布式训练
echo "Starting distributed training with 2 GPUs..."
echo "NCCL timeout set to 30 minutes"
echo "Data directory: /root/autodl-tmp/optimum-main/examples/cifar-10-batches-py"

torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    train_mor_vit_cifar10.py \
    --data_dir /root/autodl-tmp/optimum-main/examples/cifar-10-batches-py \
    --batch_size 256 \
    --epochs 100 \
    --lr 6e-5 \
    --use_pretrained \
    --enable_dora

echo "Training completed!"
