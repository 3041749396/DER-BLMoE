import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 当输入和输出维度不匹配时，调整维度
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # print('res0', x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        # print('res1',out.shape)
        out = self.bn2(self.conv2(out))
        # print('res2', out.shape)
        out += self.shortcut(x)  # 残差连接
        return F.relu(out)


class  BaseModel(nn.Module):
    def __init__(self, num_classes=100):
        super( BaseModel, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # 残差块
        self.res_block1 = ResidualBlock(32, 64, stride=2)
        self.res_block2 = ResidualBlock(64, 64, stride=2)
        self.res_block3 = ResidualBlock(64, 128, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.res_block1(out)
        
        # 检查res_block2是否为深度可分离卷积，并且输入通道数不匹配
        if hasattr(self, 'res_block2') and hasattr(self.res_block2, 'depthwise'):
            expected_channels = self.res_block2.depthwise.in_channels
            actual_channels = out.size(1)
            
            if expected_channels != actual_channels:
                # 通道数不匹配，需要调整
                if expected_channels < actual_channels:
                    # 如果期望的通道数少于实际通道数，选择前N个通道
                    out = out[:, :expected_channels, :, :]
                else:
                    # 如果期望的通道数多于实际通道数，填充额外的通道
                    padding = torch.zeros(out.size(0), expected_channels - actual_channels, 
                                         out.size(2), out.size(3), device=out.device)
                    out = torch.cat([out, padding], dim=1)
        
        out = self.res_block2(out)
        
        # 同样检查res_block3
        if hasattr(self, 'res_block3') and hasattr(self.res_block3, 'depthwise'):
            expected_channels = self.res_block3.depthwise.in_channels
            actual_channels = out.size(1)
            
            if expected_channels != actual_channels:
                if expected_channels < actual_channels:
                    out = out[:, :expected_channels, :, :]
                else:
                    padding = torch.zeros(out.size(0), expected_channels - actual_channels, 
                                         out.size(2), out.size(3), device=out.device)
                    out = torch.cat([out, padding], dim=1)
        
        out = self.res_block3(out)
        out = self.avg_pool(out)
        features = out.view(out.size(0), -1)
        out = self.fc(features)
        return out, features


    def get_compressible_layers(self):
        """返回可压缩的层列表"""
        return [
            self.conv1,
            self.res_block1.conv1, self.res_block1.conv2,
            self.res_block2.conv1, self.res_block2.conv2,
            self.res_block3.conv1, self.res_block3.conv2,
        ]
    
    def get_residual_blocks(self):
        """返回可替换为深度可分离卷积的残差块列表"""
        return [
            self.res_block1,
            self.res_block2,
            self.res_block3
        ]

    def get_block_channels(self, block_name):
        """
        获取指定残差块的输入和输出通道数
        
        参数:
            block_name: 残差块名称
        
        返回:
            (in_channels, out_channels): 输入和输出通道数的元组
        """
        if block_name == 'res_block1':
            return self.res_block1.conv1.in_channels, self.res_block1.conv2.out_channels
        elif block_name == 'res_block2':
            return self.res_block2.conv1.in_channels, self.res_block2.conv2.out_channels
        elif block_name == 'res_block3':
            return self.res_block3.conv1.in_channels, self.res_block3.conv2.out_channels
        else:
            raise ValueError(f"未知的残差块名称: {block_name}")

    def get_block_stride(self, block_name):
        """
        获取指定残差块的步长
        
        参数:
            block_name: 残差块名称
        
        返回:
            stride: 步长
        """
        if block_name == 'res_block1':
            return self.res_block1.conv1.stride[0]
        elif block_name == 'res_block2':
            return self.res_block2.conv1.stride[0]
        elif block_name == 'res_block3':
            return self.res_block3.conv1.stride[0]
        else:
            raise ValueError(f"未知的残差块名称: {block_name}") 