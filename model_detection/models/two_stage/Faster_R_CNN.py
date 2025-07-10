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
import torch
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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# model
net = fasterrcnn_resnet50_fpn(
    pretrained = True,
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT, 
    progress = False
).to(device)
net.eval();

# one class scratch+ background
num_classes = 2

# get number of input features for the classifier
in_features = net.roi_heads.box_predictor.cls_score.in_features

# replace th epre-trained head with a new one
net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
