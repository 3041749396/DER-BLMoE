import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from third_party.ajcn_src.models.base_model import BaseModel, ResidualBlock

class PrunedResidualBlock(ResidualBlock):
    """Channel-pruned residual block compatible with AJCN pruning actions.

    This is a compatibility fix for the namespaced AJCN copy used by the
    DER-BLMoE integration. The original AJCN implementation only rebuilds
    conv2 when prune_ratio_conv2 > 0. If conv1 is pruned but conv2 is not,
    conv2 still expects the original intermediate channel count, which causes
    a runtime channel mismatch. Here, conv2 is rebuilt whenever either conv1
    or conv2 changes, while preserving the original pruning semantics.
    """
    def __init__(self, in_channels, out_channels, stride=1, prune_ratio_conv1=0, prune_ratio_conv2=0):
        nn.Module.__init__(self)
        prune_ratio_conv1 = float(prune_ratio_conv1)
        prune_ratio_conv2 = float(prune_ratio_conv2)

        mid_channels = max(1, int(out_channels * (1 - prune_ratio_conv1)))
        final_channels = max(1, int(out_channels * (1 - prune_ratio_conv2)))
        self.pruned_out_channels_conv1 = mid_channels
        self.pruned_out_channels_conv2 = final_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, final_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(final_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != final_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, final_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(final_channels)
            )

class PrunedModel(BaseModel):
    """Channel-pruned AJCN model."""
    def __init__(self, num_classes=100, prune_ratios=None):
        nn.Module.__init__(self)

        if prune_ratios is None:
            prune_ratios = {
                'conv1': 0,
                'res_block1.conv1': 0, 'res_block1.conv2': 0,
                'res_block2.conv1': 0, 'res_block2.conv2': 0,
                'res_block3.conv1': 0, 'res_block3.conv2': 0
            }

        conv1_out_channels = max(1, int(32 * (1 - float(prune_ratios['conv1']))))
        self.conv1 = nn.Conv2d(3, conv1_out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_out_channels)

        self.res_block1 = PrunedResidualBlock(
            conv1_out_channels, 64, stride=2,
            prune_ratio_conv1=prune_ratios['res_block1.conv1'],
            prune_ratio_conv2=prune_ratios['res_block1.conv2']
        )
        res_block1_out_channels = self.res_block1.conv2.out_channels

        self.res_block2 = PrunedResidualBlock(
            res_block1_out_channels, 64, stride=2,
            prune_ratio_conv1=prune_ratios['res_block2.conv1'],
            prune_ratio_conv2=prune_ratios['res_block2.conv2']
        )
        res_block2_out_channels = self.res_block2.conv2.out_channels

        self.res_block3 = PrunedResidualBlock(
            res_block2_out_channels, 128, stride=2,
            prune_ratio_conv1=prune_ratios['res_block3.conv1'],
            prune_ratio_conv2=prune_ratios['res_block3.conv2']
        )
        res_block3_out_channels = self.res_block3.conv2.out_channels

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
