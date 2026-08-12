import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tqdm

def evaluate_model(model, data_loader, device):
    """
    评估模型精度
    
    参数:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 计算设备
    
    返回:
        accuracy: 精度（百分比）
    """
    model = model.to(device)  # 确保模型在正确的设备上
    model.eval()
    correct = 0
    total = 0
    
    try:
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 处理模型输出可能是元组的情况
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # 取第一个元素作为分类输出
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100.0 * correct / total
        return accuracy
    
    except Exception as e:
        print(f"评估模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0  # 出错时返回0精度

def measure_inference_time(model, input_shape=(1, 3, 32, 32), device='cuda', num_runs=100):
    """
    测量模型推理时间
    
    参数:
        model: 要评估的模型
        input_shape: 输入张量形状
        device: 计算设备
        num_runs: 运行次数
    
    返回:
        avg_time: 平均推理时间(ms)
    """
    model = model.to(device)
    model.eval()
    
    try:
        # 创建随机输入
        input_tensor = torch.randn(input_shape).to(device)
        
        # 打印输入形状，用于调试
        # print(f"推理输入形状: {input_tensor.shape}")
        
        # 预热
        with torch.no_grad():
            for _ in range(10):
                try:
                    if hasattr(model, 'forward') and 'features' in model.forward.__code__.co_varnames:
                        model(input_tensor)
                    else:
                        model(input_tensor)
                except Exception as e:
                    print(f"模型预热错误: {e}")
                    print(f"输入形状: {input_tensor.shape}")
                    raise
        
        # 测量推理时间
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                try:
                    if hasattr(model, 'forward') and 'features' in model.forward.__code__.co_varnames:
                        model(input_tensor)
                    else:
                        model(input_tensor)
                except Exception as e:
                    print(f"模型推理错误: {e}")
                    print(f"输入形状: {input_tensor.shape}")
                    raise
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        avg_time = np.mean(times)
        return avg_time
    except Exception as e:
        print(f"测量推理时间时出错: {e}")
        raise

def count_parameters(model):
    """
    计算模型参数量
    
    参数:
        model: 要评估的模型
    
    返回:
        total_params: 参数总量
    """
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params
    except Exception as e:
        print(f"计算参数量时出错: {e}")
        raise

def evaluate_compression(original_model, compressed_model, data_loader, device):
    """
    评估压缩效果
    
    参数:
        original_model: 原始模型
        compressed_model: 压缩后的模型
        data_loader: 数据加载器
        device: 计算设备
    
    返回:
        results: 包含评估结果的字典
    """
    try:
        # 获取数据集的输入形状
        for inputs, _ in data_loader:
            input_shape = inputs[0:1].shape
            print(f"数据输入形状: {input_shape}")
            break
        
        # 评估精度
        print("评估原始模型精度...")
        original_accuracy = evaluate_model(original_model, data_loader, device)
        print("评估压缩模型精度...")
        compressed_accuracy = evaluate_model(compressed_model, data_loader, device)
        
        # 计算参数量
        print("计算参数量...")
        original_params = count_parameters(original_model)
        compressed_params = count_parameters(compressed_model)
        
        # 测量推理时间
        print("测量原始模型推理时间...")
        original_time = measure_inference_time(original_model, input_shape=input_shape, device=device)
        print("测量压缩模型推理时间...")
        compressed_time = measure_inference_time(compressed_model, input_shape=input_shape, device=device)
        
        # 计算压缩比例
        accuracy_ratio = compressed_accuracy / original_accuracy
        params_ratio = compressed_params / original_params
        time_ratio = compressed_time / original_time
        
        results = {
            'original_accuracy': original_accuracy,
            'compressed_accuracy': compressed_accuracy,
            'original_params': original_params,
            'compressed_params': compressed_params,
            'original_time': original_time,
            'compressed_time': compressed_time,
            'accuracy_ratio': accuracy_ratio,
            'params_ratio': params_ratio,
            'time_ratio': time_ratio,
            'compression_ratio': 1.0 - params_ratio,
            'speedup_ratio': original_time / compressed_time
        }
        
        return results
    except Exception as e:
        print(f"评估压缩效果时出错: {e}")
        raise 