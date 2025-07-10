# -*- coding: utf-8 -*-


# ***************************************************
# * File        : Fast_FCN.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-29
# * Version     : 0.1.032916
# * Description : description
# * Link        : Semantic Segmenttation Model
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

from torchvision.models.segmentation import (
    fcn_resnet50,
    FCN_ResNet50_Weights,
    
    fcn_resnet101,
    FCN_ResNet101_Weights,
)


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


net = fcn_resnet50(weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
net.eval();




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
