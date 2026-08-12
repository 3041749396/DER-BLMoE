import torch
import torch.nn as nn
import numpy as np
import copy
import time
from rl.environment import CompressionEnv
from utils.distributed_utils import is_main_process, all_gather, reduce_dict, synchronize

class DistributedCompressionEnv(CompressionEnv):
    """
    分布式压缩环境，支持在多个GPU上并行评估模型
    """
    def __init__(self, model, train_loader, val_loader, device, rank=0, world_size=1,
                 target_accuracy=0.9, target_compression=0.8, alpha=1.0, beta=0.5, gamma=0.5):
        """
        初始化分布式压缩环境
        
        参数:
            model: 要压缩的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备
            rank: 当前进程的排名
            world_size: 总进程数
            target_accuracy: 目标精度比例（相对于原始模型）
            target_compression: 目标压缩比例
            alpha: 精度奖励权重
            beta: 模型大小奖励权重
            gamma: 推理时间奖励权重
        """
        # 调用父类初始化方法
        super().__init__(model, train_loader, val_loader, device, 
                         target_accuracy, target_compression, alpha, beta, gamma)
        
        # 分布式训练参数
        self.rank = rank
        self.world_size = world_size
        self.is_main = is_main_process(rank)
    
    def evaluate(self, model):
        """
        评估模型的精度（分布式版本）
        
        参数:
            model: 要评估的模型
        
        返回:
            accuracy: 验证集上的精度
        """
        # 如果不是分布式模式，则调用父类方法
        if self.world_size == 1:
            return super().evaluate(model)
        
        # 同步所有进程
        synchronize()
        
        model.eval()
        correct = 0
        total = 0
        
        # 计算验证集样本总数
        dataset_size = len(self.val_loader.dataset)
        
        # 为每个进程分配数据子集
        per_rank_size = dataset_size // self.world_size
        start_idx = self.rank * per_rank_size
        end_idx = start_idx + per_rank_size if self.rank < self.world_size - 1 else dataset_size
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                # 获取批次中的样本索引
                batch_start_idx = batch_idx * self.val_loader.batch_size
                
                # 如果批次在当前进程的范围外，则跳过
                if batch_start_idx >= end_idx or batch_start_idx + len(inputs) <= start_idx:
                    continue
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(targets).sum().item()
                
                correct += batch_correct
                total += targets.size(0)
        
        # 在所有进程间同步正确预测数和总样本数
        correct_tensor = torch.tensor([correct], dtype=torch.float32, device=self.device)
        total_tensor = torch.tensor([total], dtype=torch.float32, device=self.device)
        
        torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
        
        correct = correct_tensor.item()
        total = total_tensor.item()
        
        # 计算精度
        accuracy = 100.0 * correct / total if total > 0 else 0
        
        return accuracy
    
    def get_inference_time(self, model, num_trials=100):
        """
        测量模型的推理时间（分布式版本）
        
        参数:
            model: 要测量的模型
            num_trials: 测量次数
        
        返回:
            inference_time: 平均推理时间（毫秒）
        """
        # 如果只有一个进程或是主进程，则进行测量
        if self.world_size == 1 or self.is_main:
            return super().get_inference_time(model, num_trials)
        else:
            # 非主进程返回0，稍后会通过同步获取正确值
            return 0.0
    
    def step(self, action):
        """
        执行动作，更新环境状态（分布式版本）
        
        参数:
            action: 要执行的动作，[prune_action, convert_action]
        
        返回:
            next_state: 下一个状态
            reward: 获得的奖励
            done: 是否结束
        """
        # 同步所有进程，确保它们在相同的状态上执行相同的动作
        if self.world_size > 1:
            synchronize()
        
        # 调用父类的step方法
        next_state, reward, done = super().step(action)
        
        # 如果是分布式模式，同步奖励
        if self.world_size > 1:
            reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
            torch.distributed.all_reduce(reward_tensor, op=torch.distributed.ReduceOp.SUM)
            reward = reward_tensor.item() / self.world_size
        
        return next_state, reward, done
    
    def reset(self):
        """
        重置环境状态（分布式版本）
        
        返回:
            state: 初始状态
        """
        # 同步所有进程
        if self.world_size > 1:
            synchronize()
        
        # 调用父类的reset方法
        return super().reset() 