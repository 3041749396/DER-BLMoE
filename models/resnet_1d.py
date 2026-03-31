# models/resnet_1d.py
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 1D convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 1D convolution"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    """
    标准 ResNet BasicBlock 的 1D 版本：
    Conv-BN-ReLU-Conv-BN + identity
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet_1D(nn.Module):
    """
    1D ResNet 架构（18/34），轻量版：
    - 输入通道：2（IQ）
    - 基础通道数 base_width = 32（传统 ResNet 为 64）
      对比原版参数量约减少 3~4 倍
    """

    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        input_channels=2,
        base_width=32,
    ):
        super(ResNet_1D, self).__init__()

        self.inplanes = base_width
        self.conv1 = nn.Conv1d(
            input_channels,
            base_width,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 四个 stage，通道宽度分别为：
        # base_width, 2*base_width, 4*base_width, 8*base_width
        self.layer1 = self._make_layer(block, base_width, layers[0])
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建一个 stage：可选首层 stride=2 降采样，其余 stride=1。
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 下采样用于匹配通道数和长度
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 2, L]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)  # [B, base_width, ...]
        x = self.layer2(x)  # [B, 2*base_width, ...]
        x = self.layer3(x)  # [B, 4*base_width, ...]
        x = self.layer4(x)  # [B, 8*base_width, ...]

        x = self.avgpool(x)  # [B, C, 1]
        x = torch.flatten(x, 1)  # [B, C]
        x = self.fc(x)  # [B, num_classes]
        return x


def ResNet18_1D(num_classes=10, input_channels=2):
    """
    轻量 ResNet-18 1D 版本：
    - 层数：2,2,2,2
    - 基础通道数 base_width=32
    """
    return ResNet_1D(
        BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        input_channels=input_channels,
        base_width=32,
    )


def ResNet34_1D(num_classes=10, input_channels=2):
    """
    轻量 ResNet-34 1D 版本：
    - 层数：3,4,6,3
    - 同样使用 base_width=32
    """
    return ResNet_1D(
        BasicBlock,
        [3, 4, 6, 3],
        num_classes=num_classes,
        input_channels=input_channels,
        base_width=32,
    )
