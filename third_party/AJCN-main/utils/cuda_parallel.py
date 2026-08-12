import torch
import torch.nn as nn
import torch.cuda as cuda
import numpy as np

class CudaParallel:
    """
    CUDA并行计算工具类，用于在单个进程中利用多个GPU进行并行计算
    """
    def __init__(self):
        """初始化CUDA并行计算工具"""
        self.num_gpus = torch.cuda.device_count()
        self.devices = [torch.device(f"cuda:{i}") for i in range(self.num_gpus)]
        self.is_available = self.num_gpus > 0
        
        if self.is_available:
            print(f"检测到 {self.num_gpus} 个CUDA设备:")
            for i in range(self.num_gpus):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    
    def parallelize_model(self, model):
        """
        将模型并行化，使用DataParallel封装
        
        参数:
            model: 要并行化的模型
        
        返回:
            并行化后的模型
        """
        if self.is_available and self.num_gpus > 1:
            return nn.DataParallel(model.cuda())
        else:
            return model.cuda() if self.is_available else model
    
    def parallelize_batch(self, batch, batch_size=None):
        """
        将数据批次分成多个子批次，分配到多个GPU上
        
        参数:
            batch: 数据批次，可以是张量、列表或字典
            batch_size: 每个子批次的大小，如果为None则自动计算
        
        返回:
            子批次列表，每个子批次对应一个GPU
        """
        if not self.is_available or self.num_gpus <= 1:
            return [batch]
        
        # 如果是张量
        if isinstance(batch, torch.Tensor):
            total_size = batch.size(0)
            if batch_size is None:
                batch_size = (total_size + self.num_gpus - 1) // self.num_gpus
            
            sub_batches = []
            for i in range(0, total_size, batch_size):
                end = min(i + batch_size, total_size)
                sub_batch = batch[i:end].cuda(self.devices[len(sub_batches) % self.num_gpus])
                sub_batches.append(sub_batch)
            
            return sub_batches
        
        # 如果是列表
        elif isinstance(batch, list):
            total_size = len(batch)
            if batch_size is None:
                batch_size = (total_size + self.num_gpus - 1) // self.num_gpus
            
            sub_batches = []
            for i in range(0, total_size, batch_size):
                end = min(i + batch_size, total_size)
                sub_batch = batch[i:end]
                sub_batches.append(sub_batch)
            
            return sub_batches
        
        # 如果是字典
        elif isinstance(batch, dict):
            keys = list(batch.keys())
            if not keys:
                return [batch]
            
            # 假设所有值的长度相同
            total_size = len(batch[keys[0]])
            if batch_size is None:
                batch_size = (total_size + self.num_gpus - 1) // self.num_gpus
            
            sub_batches = []
            for i in range(0, total_size, batch_size):
                end = min(i + batch_size, total_size)
                sub_batch = {k: batch[k][i:end] for k in keys}
                sub_batches.append(sub_batch)
            
            return sub_batches
        
        # 不支持的类型
        else:
            raise TypeError(f"不支持的批次类型: {type(batch)}")
    
    def parallel_forward(self, model, inputs):
        """
        并行前向传播
        
        参数:
            model: 模型
            inputs: 输入数据
        
        返回:
            前向传播的结果
        """
        if not self.is_available:
            return model(inputs)
        
        if isinstance(model, nn.DataParallel):
            # 如果模型已经是DataParallel，直接使用
            return model(inputs.cuda())
        else:
            # 否则，手动分配到多个GPU
            sub_batches = self.parallelize_batch(inputs)
            results = []
            
            for i, sub_batch in enumerate(sub_batches):
                device = self.devices[i % self.num_gpus]
                with torch.cuda.device(device):
                    model_on_device = model.cuda(device)
                    results.append(model_on_device(sub_batch))
            
            # 合并结果
            if isinstance(results[0], torch.Tensor):
                return torch.cat(results, dim=0)
            else:
                # 如果结果不是张量，需要根据具体情况处理
                return results
    
    def parallel_evaluate(self, model, data_loader, criterion=None):
        """
        并行评估模型
        
        参数:
            model: 要评估的模型
            data_loader: 数据加载器
            criterion: 评估准则，如果为None则计算准确率
        
        返回:
            评估结果
        """
        if not self.is_available:
            device = torch.device("cpu")
            model = model.to(device)
        else:
            if isinstance(model, nn.DataParallel):
                device = torch.device("cuda")
            else:
                device = torch.device("cuda:0")
                model = model.to(device)
        
        model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                
                if criterion is not None:
                    loss = criterion(outputs, targets)
                    loss_sum += loss.item() * inputs.size(0)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        if criterion is not None:
            avg_loss = loss_sum / total if total > 0 else 0.0
            return accuracy, avg_loss
        else:
            return accuracy
    
    def parallel_inference_time(self, model, input_shape, num_trials=100):
        """
        测量模型的并行推理时间
        
        参数:
            model: 要测量的模型
            input_shape: 输入形状
            num_trials: 测量次数
        
        返回:
            平均推理时间（毫秒）
        """
        if not self.is_available:
            return self._inference_time_cpu(model, input_shape, num_trials)
        
        if isinstance(model, nn.DataParallel):
            model.eval()
            device = torch.device("cuda")
        else:
            model = model.cuda()
            model.eval()
            device = torch.device("cuda:0")
        
        # 创建随机输入
        dummy_input = torch.randn(input_shape, device=device)
        
        # 预热
        for _ in range(10):
            _ = model(dummy_input)
        
        torch.cuda.synchronize()
        
        # 测量时间
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        for _ in range(num_trials):
            _ = model(dummy_input)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / num_trials
        
        return elapsed_time
    
    def _inference_time_cpu(self, model, input_shape, num_trials=100):
        """
        在CPU上测量模型的推理时间
        
        参数:
            model: 要测量的模型
            input_shape: 输入形状
            num_trials: 测量次数
        
        返回:
            平均推理时间（毫秒）
        """
        import time
        
        model = model.cpu()
        model.eval()
        
        # 创建随机输入
        dummy_input = torch.randn(input_shape)
        
        # 预热
        for _ in range(10):
            _ = model(dummy_input)
        
        # 测量时间
        start_time = time.time()
        for _ in range(num_trials):
            _ = model(dummy_input)
        elapsed_time = (time.time() - start_time) / num_trials * 1000  # 转换为毫秒
        
        return elapsed_time 