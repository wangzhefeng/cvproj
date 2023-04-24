# -*- coding: utf-8 -*-


# ***************************************************
# * File        : faster_rcnn_model.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-01-19
# * Version     : 0.1.011916
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
_path = os.path.abspath(os.path.dirname(__file__))
if os.path.join(_path, "..") not in sys.path:
    sys.path.append(os.path.join(_path, ".."))

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import mobilenet_v2
from torchvision.ops import MultiScaleRoIAlign
# utils
import transforms as T
from engine import train_one_epoch, evaluate
import utils
# data set
from PennFudanDataset import PennFudanDataset


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

"""
# ------------------------------
# 从 COCO 载入预训练模型
# ------------------------------
model = fasterrcnn_resnet50_fpn(weights = "DEFAULT")
# ------------------------------
# 微调预训练模型 
# ------------------------------
# 1 class(person) + background
num_classes = 2
# get number of input features for classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# ------------------------------
# 修改模型增加一个骨干(支柱)
# ------------------------------
# 载入一个分类预训练模型，并返回其特征
backbone = mobilenet_v2(weights = "DEFAULT").features
# FasterRCNN 需要知道一个 backbone 的输出 通道数量
backbone.out_channels = 1280
# 让 RPN 在每个空间生成 5 x 3 个锚点(anchors)位置，有 5 种不同的大小和 3 种不同的方位比率
anchor_generator = AnchorGenerator(
    sizes = ((32, 64, 128, 256, 512),),
    aspect_ratios = ((0.5, 1.0, 2.0),)
)
# 定义用来执行感兴趣区域裁剪的特征图以及重新缩放后裁剪的大小
roi_pooler = MultiScaleRoIAlign(
    featmap_names = ["0"],
    output_size = 7,
    sampling_ratio = 2
)
# ------------------------------
# model
# ------------------------------
model = FasterRCNN(
    backbone = backbone,
    num_classes = num_classes,
    rpn_anchor_generator = anchor_generator,
    box_roi_pool = roi_pooler,
)
"""
# ------------------------------
# 分割模型实例
# ------------------------------
def get_model_instance_segmentation(num_classes):
    # pre-trained segmentation model on COCO
    model = maskrcnn_resnet50_fpn(pretrained = True)
    # input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace pre-trained head
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # input features for mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # number of hidden layers
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)




# 测试代码 main 函数
def main():
    # device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # classes
    num_classes = 2
    # ------------------------------
    # data
    # ------------------------------
    root = "/Users/zfwang/learn/ml/dl/dlproj/src_pytorch/image_object_detection/PennFudanPed"
    dataset = PennFudanDataset(root, get_transform(train = True))
    dataset_test = PennFudanDataset(root, get_transform(train = False))
    # split dataset in train and test
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_set = torch.utils.data.Subset(dataset_test, indices[-50:])
    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = 2,
        shuffle = True,
        num_workers = 4,
        collate_fn = utils.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size = 1,
        shuffle = False,
        num_workers = 4,
        collate_fn = utils.collate_fn
    )
    # ------------------------------
    # model
    # ------------------------------
    model = get_model_instance_segmentation(num_classes)
    # device
    model.to(device)
    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr = 0.005,
        momentum = 0.9,
        weight_decay = 0.0005
    )
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = 3,
        gamma = 0.1
    )
    # ------------------------------
    # model train
    # ------------------------------
    num_epochs = 10
    for epoch in range(num_epochs):
        # train
        train_one_epoch(
            model, 
            optimizer, 
            data_loader, 
            device, 
            epoch, 
            print_freq = 10
        )
        # update learning rate
        lr_scheduler.step()
        # evaluate
        evaluate(model, data_loader_test, device = device)

    print("That's it!")




if __name__ == "__main__":
    main()

