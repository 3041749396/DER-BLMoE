# models/cnn_1d.py
import torch
import torch.nn as nn


class InvertedResidual1D(nn.Module):
    """
    MobileNetV2 里的 Inverted Residual Block 的 1D 版本：
    - 1x1 pw (expand)
    - 3x3 dw (depthwise)
    - 1x1 pw-linear (project)
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual1D, self).__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            # pw
            layers.extend([
                nn.Conv1d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        # dw
        layers.extend([
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        # pw-linear
        layers.extend([
            nn.Conv1d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(oup),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_1D(nn.Module):
    """
    轻量 MobileNetV2 风格 1D CNN：
    - 输入通道: 2 (IQ)
    - 大量 depthwise separable conv，参数非常少
    - 自适应池化到长度 1，再线性分类

    参数量大约在 O(10^5 - 10^6) 级，适合作为轻量 DL baseline。
    """

    def __init__(self, num_classes: int = 10, input_len: int = 128,
                 width_mult: float = 0.5):
        """
        num_classes : 分类类别数
        input_len   : 序列长度（保留此参数以兼容旧接口，内部只用于注释，不依赖具体长度）
        width_mult  : 通道宽度缩放因子，默认 0.5 进一步减小参数量
        """
        super(MobileNetV2_1D, self).__init__()

        # t, c, n, s 配置（来自 MobileNetV2，做了轻量化缩减）
        # t: expand ratio, c: output channels, n: num blocks, s: stride
        cfgs = [
            # t,  c,  n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 3, 2],
            [6,  96, 2, 1],
        ]

        input_channel = int(32 * width_mult)
        last_channel = int(128 * width_mult)

        # Stem: 3x3 conv
        layers = [
            nn.Conv1d(2, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(input_channel),
            nn.ReLU6(inplace=True),
        ]

        # Inverted residual blocks
        block = InvertedResidual1D
        for t, c, n, s in cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # Last 1x1 conv
        layers.extend([
            nn.Conv1d(input_channel, last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(last_channel),
            nn.ReLU6(inplace=True),
        ])

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: [B, 2, L]
        x = self.features(x)
        x = self.avgpool(x)   # [B, C, 1]
        x = x.view(x.size(0), -1)  # [B, C]
        x = self.classifier(x)
        return x
