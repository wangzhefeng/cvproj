# -*- coding: utf-8 -*-


# ***************************************************
# * File        : ResNet101.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-29
# * Version     : 0.1.032911
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
from torch import nn
from torchvision.models import (
    resnet101,
    ResNet101_Weights,
    resnext101_32x8d,
    ResNeXt101_32X8D_Weights,
    resnext101_64x4d,
    ResNeXt101_64X4D_Weights,
    Wide_ResNet101_2_Weights,
    wide_resnet101_2,
)


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


net = resnet101(weights = ResNet101_Weights.DEFAULT, progress = False)
net.eval()






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
