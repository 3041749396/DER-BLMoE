import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator

def plot_training_curves(rewards, accuracies, params_ratios, time_ratios, save_path=None):
    """
    绘制训练曲线
    
    参数:
        rewards: 奖励列表
        accuracies: 精度列表
        params_ratios: 参数比例列表
        time_ratios: 时间比例列表
        save_path: 保存路径
    """
    plt.figure(figsize=(15, 10))
    
    # 绘制奖励曲线
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward')
    plt.grid(True)
    
    # 绘制精度曲线
    plt.subplot(2, 2, 2)
    plt.plot(accuracies)
    plt.xlabel('Episode')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy')
    plt.grid(True)
    
    # 绘制参数比例曲线
    plt.subplot(2, 2, 3)
    plt.plot(params_ratios)
    plt.xlabel('Episode')
    plt.ylabel('Parameters Ratio')
    plt.title('Parameters Compression Ratio')
    plt.grid(True)
    
    # 绘制时间比例曲线
    plt.subplot(2, 2, 4)
    plt.plot(time_ratios)
    plt.xlabel('Episode')
    plt.ylabel('Inference Time Ratio')
    plt.title('Inference Time Ratio')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_compression_results(results, save_path=None):
    """
    绘制压缩结果
    
    参数:
        results: 压缩结果字典
        save_path: 保存路径
    """
    # 提取结果
    original_accuracy = results['original_accuracy']
    compressed_accuracy = results['compressed_accuracy']
    original_params = results['original_params']
    compressed_params = results['compressed_params']
    original_time = results['original_time']
    compressed_time = results['compressed_time']
    
    # 创建图表
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # 绘制精度对比
    ax[0].bar(['Original', 'Compressed'], [original_accuracy, compressed_accuracy])
    ax[0].set_ylabel('Accuracy (%)')
    ax[0].set_title('Model Accuracy')
    ax[0].grid(axis='y')
    
    # 绘制参数量对比
    ax[1].bar(['Original', 'Compressed'], [original_params/1e6, compressed_params/1e6])
    ax[1].set_ylabel('Parameters (M)')
    ax[1].set_title('Model Size')
    ax[1].grid(axis='y')
    
    # 绘制推理时间对比
    ax[2].bar(['Original', 'Compressed'], [original_time, compressed_time])
    ax[2].set_ylabel('Inference Time (ms)')
    ax[2].set_title('Inference Time')
    ax[2].grid(axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_layer_compression(prune_ratios, convert_blocks, save_path=None):
    """
    绘制各层压缩情况
    
    参数:
        prune_ratios: 每层的剪枝比例（列表）
        convert_blocks: 每层是否转换为深度可分离卷积（列表）
        save_path: 保存路径
    """
    # 定义层名称
    layers = [
        'conv1',
        'res_block1',
        'res_block2', 
        'res_block3'
    ]
    
    # 确保列表长度匹配
    if len(prune_ratios) != len(layers):
        print(f"警告: 剪枝比例列表长度 ({len(prune_ratios)}) 与层数 ({len(layers)}) 不匹配")
        # 如果列表长度不匹配，使用可用的数据
        prune_ratios = prune_ratios[:len(layers)] if len(prune_ratios) > len(layers) else prune_ratios + [0] * (len(layers) - len(prune_ratios))
    
    if len(convert_blocks) != len(layers):
        print(f"警告: 转换块列表长度 ({len(convert_blocks)}) 与层数 ({len(layers)}) 不匹配")
        # 如果列表长度不匹配，使用可用的数据
        convert_blocks = convert_blocks[:len(layers)] if len(convert_blocks) > len(layers) else convert_blocks + [0] * (len(layers) - len(convert_blocks))
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制剪枝比例
    ax1.bar(layers, prune_ratios, color='skyblue')
    ax1.set_title('各层剪枝比例')
    ax1.set_xlabel('层名称')
    ax1.set_ylabel('剪枝比例')
    ax1.set_ylim(0, 1)
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(prune_ratios):
        ax1.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    # 绘制是否转换为深度可分离卷积
    ax2.bar(layers, convert_blocks, color='salmon')
    ax2.set_title('各层是否转换为深度可分离卷积')
    ax2.set_xlabel('层名称')
    ax2.set_ylabel('是否转换 (0=否, 1=是)')
    ax2.set_ylim(0, 1.2)
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(convert_blocks):
        ax2.text(i, v + 0.05, str(v), ha='center')
    
    plt.tight_layout()
    
    # 保存图形
    if save_path:
        plt.savefig(save_path)
        print(f"图形已保存至: {save_path}")
    
    plt.close()

def visualize_network_structure(model, input_size=(1, 3, 32, 32), save_path=None):
    """
    可视化网络结构
    
    参数:
        model: 要可视化的模型
        input_size: 输入张量大小
        save_path: 保存路径
    """
    try:
        from torchviz import make_dot
        
        # 创建输入张量
        x = torch.randn(input_size)
        
        # 前向传播
        y, _ = model(x)
        
        # 创建计算图
        dot = make_dot(y, params=dict(model.named_parameters()))
        
        # 设置图形属性
        dot.attr('graph', rankdir='TB')  # 从上到下的布局
        dot.attr('node', shape='box')
        
        # 保存或显示图形
        if save_path:
            dot.render(save_path, format='png')
        else:
            dot.view()
    
    except ImportError:
        print("请安装torchviz: pip install torchviz")
        print("并安装Graphviz: https://graphviz.org/download/") 