# -*- coding: utf-8 -*-


# ***************************************************
# * File        : DeepLab.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-29
# * Version     : 0.1.032916
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
from torchvision.models.segmentation import (
    deeplabv3,
    DeepLabV3,
    
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
    
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
    
    deeplabv3_mobilenet_v3_large,
    DeepLabV3_MobileNet_V3_Large_Weights,
)


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
