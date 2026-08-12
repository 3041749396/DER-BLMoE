#!/bin/bash

# 设置Python路径
PYTHON="python"

# 设置CUDA可见设备（列出所有GPU设备ID）
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置CUDA环境
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 运行CUDA并行训练
$PYTHON train_cuda_parallel_rl_compression.py --config configs/config.yaml

echo "CUDA并行训练完成！" 