@echo off
setlocal

:: 设置Python路径
set PYTHON=python

:: 设置CUDA可见设备（列出所有GPU设备ID）
set CUDA_VISIBLE_DEVICES=0,1,2,3

:: 设置CUDA环境
set CUDA_DEVICE_ORDER=PCI_BUS_ID

:: 运行CUDA并行训练
%PYTHON% train_cuda_parallel_rl_compression.py --config configs/config.yaml

echo CUDA并行训练完成！

pause 