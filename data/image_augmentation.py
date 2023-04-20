# -*- coding: utf-8 -*-


# ***************************************************
# * File        : image_augmentation.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-27
# * Version     : 0.1.032717
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision import datasets
from d2l import torch as d2l


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# 原始图像
d2l.set_figsize()
img = d2l.Image.open("../img/cat1.jpg")
d2l.plt.imshow(img);

def apply(img, aug, num_rows = 2, num_cols = 4, scale = 1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale = scale)

# 翻转
apply(img, transforms.RandomHorizontalFlip())
apply(img, transforms.RandomVerticalFlip())

# 裁剪
shape_aug = transforms.RandomResizedCrop(
    (200, 200),
    scale = (0.1, 1),
    ratio = (0.5, 2),
)
apply(img, shape_aug)

# 改变颜色
apply(img, transforms.ColorJitter(
    brightness = 0.5, 
    contrast = 0,
    saturation = 0,
    hue = 0,
))

apply(img, transforms.ColorJitter(
    brightness = 0,
    contrast = 0,
    saturation = 0,
    hue = 0.5
))

apply(img, transforms.ColorJitter(
    brightness = 0.5,
    contrast = 0.5, 
    saturation = 0.5,
    hue = 0.5,
))

# 使用图像增强进行训练
all_images = datasets.CIFAR10(train = True, root = "../data", download = True)
d2l.show_images(
    [all_images[i][0] for i in range(32)], 
    4, 
    8, 
    scale = 0.8
)








# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
