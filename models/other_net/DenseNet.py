# -*- coding: utf-8 -*-

# ***************************************************
# * File        : DenseNet.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091422
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch import nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def conv_block(input_channels, num_channels):
    block = nn.Sequential(
        nn.BatchNorm2d(input_channels), 
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size = 3, padding = 1),
    )

    return block


class DenseBlock(nn.Module):

    def __init__(self, num_convs, input_channels, num_channels) -> None:
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)
    
    def forward(self, x):
        for blk in self.net:
            Y = blk(X)
            # 连接通道维度上每个块的输入和输出
            X = torch.cat((X, Y), dim = 1)
        return X


def transition_block(input_channels, num_channels):
    """
    过渡层
    """
    block = nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size = 1),
        nn.AvgPool2d(kernel_size = 2, stride = 2)
    )

    return block

# ------------------------------
# model training
# ------------------------------
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)

blk = transition_block(23, 10)
print(blk(Y).shape)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
