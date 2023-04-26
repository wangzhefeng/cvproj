# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Faster_R_CNN.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-29
# * Version     : 0.1.032914
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys

import torch
from torch import nn
from torchvision.transforms.functional import pil_to_tensor
from torchvision.models.detection import (
    faster_rcnn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


net = fasterrcnn_resnet50_fpn(
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT, 
    progress = False
)
net.eval();




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
