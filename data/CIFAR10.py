# -*- coding: utf-8 -*-


# ***************************************************
# * File        : CIFAR10.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-23
# * Version     : 0.1.032308
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
from torchvision import datasets


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data
def get_dataset(train_transforms, test_transforms, valid_transforms = None):
    train_dataset = datasets.CIFAR10(
        root = "./data/",
        train = True,
        transform = train_transforms,
        download = True
    )
    test_dataset = datasets.CIFAR10(
        root = "./data/",
        train = False,
        transform = test_transforms,
        download = True,
    )
    if valid_transforms:
        valid_dataset = datasets.CIFAR10(
            root = "./data/",
            train = True,
            transform = valid_transforms,
            download = True
        )
        return train_dataset, test_dataset, valid_dataset
    else:
        return train_dataset, test_dataset


# data loader
def get_dataloader(train_dataset,  
                   test_dataset, 
                   batch_size, 
                   train_sampler = None,
                   valid_dataset = None,
                   valid_sampler = None):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        sampler = train_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
    )
    if valid_dataset:
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size = batch_size,
            sampler = valid_sampler,
        )
        return train_loader, test_loader, valid_loader
    else:
        return train_loader, test_loader




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
