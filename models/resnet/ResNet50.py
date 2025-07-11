# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ResNet50.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-15
# * Version     : 1.0.091504
# * Description : description
# * Link        : https://github.com/lyhue1991/torchkeras/blob/master/torchkeras/models/resnet.py
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = in_channel, 
            out_channels = out_channel,
            kernel_size = 1, 
            stride = 1, 
            bias = False
        )  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(
            in_channels = out_channel, 
            out_channels = out_channel, 
            kernel_size = 3, 
            stride = stride, 
            bias = False, 
            padding = 1
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(
            in_channels = out_channel, 
            out_channels = out_channel * self.expansion,
            kernel_size = 1, 
            stride = 1, 
            bias = False
        )  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes = 1000, include_top = True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(
            in_channles = 3, 
            out_channels = self.in_channel, 
            kernel_size = 7, 
            stride = 2,
            padding = 3, 
            bias = False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride = 2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')

    def _make_layer(self, block, channel, block_num, stride = 1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channel, channel, downsample = downsample, stride = stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


class ResNet50(ResNet):

    def __init__(self, num_classes = 1000, include_top = True):
        super().__init__(
            Bottleneck, 
            [3, 4, 6, 3], 
            num_classes = num_classes, 
            include_top = include_top
        )


def resnet50(num_classes = 1000, include_top = True):
    net = ResNet(
        Bottleneck, 
        [3, 4, 6, 3], 
        num_classes = num_classes, 
        include_top = include_top
    )

    return net




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
