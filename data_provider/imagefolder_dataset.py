# -*- coding: utf-8 -*-

# ***************************************************
# * File        : imagefolder_dataset.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091417
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
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


"""
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
