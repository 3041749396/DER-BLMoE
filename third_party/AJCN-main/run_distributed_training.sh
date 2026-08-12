#!/bin/bash

# 设置Python路径
PYTHON="python"

# 设置CUDA可见设备
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际GPU数量修改

# 设置OMP线程数
export OMP_NUM_THREADS=1

# 运行分布式训练
$PYTHON train_distributed_rl_compression.py --config configs/distributed_config.yaml

echo "分布式训练完成！" 