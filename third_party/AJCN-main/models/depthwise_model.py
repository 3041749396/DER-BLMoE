import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel, ResidualBlock

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积实现"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积 - 每个输入通道单独卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        # 逐点卷积 - 1x1卷积用于通道混合
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, 
            stride=1, padding=0, bias=False
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DepthwiseSeparableResidualBlock(nn.Module):
    """使用深度可分离卷积的残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableResidualBlock, self).__init__()
        # 使用深度可分离卷积替代标准卷积
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 快捷连接也可以使用深度可分离卷积
            self.shortcut = nn.Sequential(
                DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DepthwiseSeparableModel(BaseModel):
    """使用深度可分离卷积的模型"""
    def __init__(self, num_classes=100, blocks_to_convert=None):
        super(BaseModel, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 残差块
        self.res_block1 = ResidualBlock(32, 64, stride=2)
        self.res_block2 = ResidualBlock(64, 64, stride=2)
        self.res_block3 = ResidualBlock(64, 128, stride=2)
        
        # 全局平均池化和全连接层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        
        # 如果指定了要转换的块，则进行转换
        if blocks_to_convert is None:
            blocks_to_convert = []
        
        # 转换指定的残差块为深度可分离卷积残差块
        if 'res_block1' in blocks_to_convert:
            self.res_block1 = DepthwiseSeparableResidualBlock(32, 64, stride=2)
        
        if 'res_block2' in blocks_to_convert:
            self.res_block2 = DepthwiseSeparableResidualBlock(64, 64, stride=2)
        
        if 'res_block3' in blocks_to_convert:
            self.res_block3 = DepthwiseSeparableResidualBlock(64, 128, stride=2)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.avg_pool(out)
        features = torch.flatten(out, 1)
        out = self.fc(features)
        return out, features 