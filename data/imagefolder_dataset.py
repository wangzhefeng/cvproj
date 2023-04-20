# -*- coding: utf-8 -*-


# ***************************************************
# * File        : dir_dataset.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-26
# * Version     : 0.1.032622
# * Description : create cifar2 dataset
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


"""
# 图片
img = Image.open("./data/cat.jpeg")
# 随机数值翻转
transforms.RandomVerticalFlip()(img)
# 随机旋转
transforms.RandomRotation(45)(img)

* ./cifar2
    - train
        - img1.png
        - img2.png
    - test
        - img1.png
        - img2.png
"""

# ------------------------------
# data
# ------------------------------
# transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomRotation(45),  # 随机在 45 角度内旋转
    transforms.ToTensor(),  # 转换成张量
])
transform_valid = transforms.Compose([
    transforms.ToTensor()
])
def transform_label(x):
    return torch.tensor([x]).float()

# dataset
train_dataset = datasets.ImageFolder(
    root = "./cifar2/train/",
    train = True,
    transform = transform_train,
    target_transform = transform_label,
)
valid_dataset = datasets.ImageFolder(
    root = "./cifar2/test/",
    train = False,
    transform = transform_valid,
    target_transform = transform_label,
)
print(train_dataset.class_to_idx)

# dataloader
train_dataloader = DataLoader(
    train_dataset, 
    batch_size = 50, 
    shuffle = True,
)
valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size = 50, 
    shuffle = False,
)

# test
for features, labels in train_dataloader:
    print(features.shape)
    print(labels.shape)
    break




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
