import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.base_model import BaseModel, ResidualBlock

class PrunedResidualBlock(ResidualBlock):
    """可剪枝的残差块"""
    def __init__(self, in_channels, out_channels, stride=1, prune_ratio_conv1=0, prune_ratio_conv2=0):
        super(PrunedResidualBlock, self).__init__(in_channels, out_channels, stride)
        
        # 计算剪枝后的通道数
        self.pruned_out_channels_conv1 = max(1, int(out_channels * (1 - prune_ratio_conv1)))
        self.pruned_out_channels_conv2 = max(1, int(out_channels * (1 - prune_ratio_conv2)))
        
        # 重新定义剪枝后的卷积层
        if prune_ratio_conv1 > 0:
            self.conv1 = nn.Conv2d(in_channels, self.pruned_out_channels_conv1, 
                                   kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.pruned_out_channels_conv1)
        
        if prune_ratio_conv2 > 0:
            # 注意：conv2的输入通道数需要匹配conv1的输出通道数
            self.conv2 = nn.Conv2d(self.pruned_out_channels_conv1, self.pruned_out_channels_conv2, 
                                   kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(self.pruned_out_channels_conv2)
            
            # 如果输出通道数改变，需要更新shortcut
            if stride != 1 or in_channels != self.pruned_out_channels_conv2:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, self.pruned_out_channels_conv2, 
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.pruned_out_channels_conv2)
                )

class PrunedModel(BaseModel):
    """通道剪枝后的模型"""
    def __init__(self, num_classes=100, prune_ratios=None):
        super(BaseModel, self).__init__()
        
        # 默认不剪枝
        if prune_ratios is None:
            prune_ratios = {
                'conv1': 0,
                'res_block1.conv1': 0, 'res_block1.conv2': 0,
                'res_block2.conv1': 0, 'res_block2.conv2': 0,
                'res_block3.conv1': 0, 'res_block3.conv2': 0
            }
        
        # 计算剪枝后的通道数
        conv1_out_channels = max(1, int(32 * (1 - prune_ratios['conv1'])))
        
        # 定义剪枝后的第一个卷积层
        self.conv1 = nn.Conv2d(3, conv1_out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_out_channels)
        
        # 定义剪枝后的残差块
        self.res_block1 = PrunedResidualBlock(
            conv1_out_channels, 64, stride=2,
            prune_ratio_conv1=prune_ratios['res_block1.conv1'],
            prune_ratio_conv2=prune_ratios['res_block1.conv2']
        )
        
        # 获取res_block1的输出通道数
        res_block1_out_channels = max(1, int(64 * (1 - prune_ratios['res_block1.conv2'])))
        
        self.res_block2 = PrunedResidualBlock(
            res_block1_out_channels, 64, stride=2,
            prune_ratio_conv1=prune_ratios['res_block2.conv1'],
            prune_ratio_conv2=prune_ratios['res_block2.conv2']
        )
        
        # 获取res_block2的输出通道数
        res_block2_out_channels = max(1, int(64 * (1 - prune_ratios['res_block2.conv2'])))
        
        self.res_block3 = PrunedResidualBlock(
            res_block2_out_channels, 128, stride=2,
            prune_ratio_conv1=prune_ratios['res_block3.conv1'],
            prune_ratio_conv2=prune_ratios['res_block3.conv2']
        )
        
        # 获取res_block3的输出通道数
        res_block3_out_channels = max(1, int(128 * (1 - prune_ratios['res_block3.conv2'])))
        
        # 全局平均池化和全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(res_block3_out_channels, num_classes)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.avg_pool(out)
        features = torch.flatten(out, 1)
        out = self.fc(features)
        return out, features 