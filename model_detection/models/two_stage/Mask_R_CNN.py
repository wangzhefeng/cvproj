# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Mask_R_CNN_pretrain.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-29
# * Version     : 0.1.032916
# * Description : description
# * Link        : Instance Segmentation Model
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import torch
import torchvision
from torchvision.models.detection import (
    mask_rcnn,
    MaskRCNN,
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


net = maskrcnn_resnet50_fpn(weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
net.eval();




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
