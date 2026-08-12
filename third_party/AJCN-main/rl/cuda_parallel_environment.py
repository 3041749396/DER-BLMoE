import torch
import torch.nn as nn
import numpy as np
import copy
import time
from rl.environment import CompressionEnv
from utils.cuda_parallel import CudaParallel

class CudaParallelCompressionEnv(CompressionEnv):
    """
    CUDA并行压缩环境，支持在多个GPU上并行评估模型
    """
    def __init__(self, model, train_loader, val_loader, device, 
                 target_accuracy=0.9, target_compression=0.8, alpha=1.0, beta=0.5, gamma=0.5):
        """
        初始化CUDA并行压缩环境
        
        参数:
            model: 要压缩的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备
            target_accuracy: 目标精度比例（相对于原始模型）
            target_compression: 目标压缩比例
            alpha: 精度奖励权重
            beta: 模型大小奖励权重
            gamma: 推理时间奖励权重
        """
        # 先初始化CUDA并行工具
        self.cuda_parallel = CudaParallel()
        
        # 如果有多个GPU可用，使用并行版本的device
        if self.cuda_parallel.is_available and self.cuda_parallel.num_gpus > 0:
            device = torch.device("cuda:0")
        
        # 调用父类初始化方法
        super().__init__(model, train_loader, val_loader, device, 
                         target_accuracy, target_compression, alpha, beta, gamma)
        
        # 如果有多个GPU可用，并行化模型
        if self.cuda_parallel.is_available and self.cuda_parallel.num_gpus > 1:
            self.original_model = self.cuda_parallel.parallelize_model(self.original_model)
            self.current_model = self.cuda_parallel.parallelize_model(self.current_model)
            print(f"已启用CUDA并行模式，使用 {self.cuda_parallel.num_gpus} 个GPU")
    
    def evaluate(self, model):
        """
        评估模型的精度（CUDA并行版本）
        
        参数:
            model: 要评估的模型
        
        返回:
            accuracy: 验证集上的精度
        """
        # 使用CUDA并行工具评估模型
        if self.cuda_parallel.is_available and self.cuda_parallel.num_gpus > 1:
            # 确保模型使用DataParallel包装
            if not isinstance(model, nn.DataParallel):
                model = self.cuda_parallel.parallelize_model(model)
            
            return self.cuda_parallel.parallel_evaluate(model, self.val_loader)
        else:
            # 如果没有多个GPU，调用父类方法
            return super().evaluate(model)
    
    def get_inference_time(self, model, num_trials=100):
        """
        测量模型的推理时间（CUDA并行版本）
        
        参数:
            model: 要测量的模型
            num_trials: 测量次数
        
        返回:
            inference_time: 平均推理时间（毫秒）
        """
        # 确定输入形状
        for inputs, _ in self.val_loader:
            input_shape = inputs.shape
            break
        
        # 使用CUDA并行工具测量推理时间
        if self.cuda_parallel.is_available:
            # 确保模型已并行化
            if self.cuda_parallel.num_gpus > 1 and not isinstance(model, nn.DataParallel):
                model = self.cuda_parallel.parallelize_model(model)
            
            return self.cuda_parallel.parallel_inference_time(model, input_shape, num_trials)
        else:
            # 如果没有GPU，调用父类方法
            return super().get_inference_time(model, num_trials)
    
    def apply_compression(self, model, prune_ratios, convert_blocks):
        """
        应用压缩策略到模型（CUDA并行版本）
        
        参数:
            model: 要压缩的模型
            prune_ratios: 各层的剪枝比例
            convert_blocks: 各层是否转换为深度可分离卷积
        
        返回:
            compressed_model: 压缩后的模型
        """
        # 如果模型是DataParallel，获取内部模型
        if isinstance(model, nn.DataParallel):
            original_model = model.module
        else:
            original_model = model
        
        # 调用父类方法压缩模型
        compressed_model = super().apply_compression(original_model, prune_ratios, convert_blocks)
        
        # 如果原模型是DataParallel，将压缩后的模型也并行化
        if isinstance(model, nn.DataParallel) and self.cuda_parallel.is_available:
            compressed_model = self.cuda_parallel.parallelize_model(compressed_model)
        
        return compressed_model
    
    def step(self, action):
        """
        执行动作，更新环境状态（CUDA并行版本）
        
        参数:
            action: 要执行的动作，[prune_action, convert_action]
        
        返回:
            next_state: 下一个状态
            reward: 获得的奖励
            done: 是否结束
        """
        # 调用父类的step方法
        next_state, reward, done = super().step(action)
        
        # 如果完成一个完整的episode，确保当前模型是并行的
        if done and self.cuda_parallel.is_available and self.cuda_parallel.num_gpus > 1:
            if not isinstance(self.current_model, nn.DataParallel):
                self.current_model = self.cuda_parallel.parallelize_model(self.current_model)
        
        return next_state, reward, done 