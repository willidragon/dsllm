#!/bin/bash

# Activate conda environment
source ~/miniconda/etc/profile.d/conda.sh
conda activate sensorllm-flash-attn

# DistributedDataParallel Launch Script for Enhancement Training
# This script launches the enhancement training on 2 A100 GPUs using DDP

echo "🚀 Launching DistributedDataParallel Enhancement Training on 2 A100 GPUs"
echo "=================================================================="

# Set environment variables for better performance
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=0

# Launch training with python3 -m torch.distributed.run (PyTorch distributed launcher)
python3 -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=12355 \
    run_data_enhancement.py

echo "🏁 DDP Training completed!" 