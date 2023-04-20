# -*- coding: utf-8 -*-


# ***************************************************
# * File        : FashionMNIST.py
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
from torchvision import transforms


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


train_transform = transforms.ToTensor()
test_transform = transforms.ToTensor()
target_transform = transforms.Lambda(
    lambda y: torch.zeros(10, dtype = torch.float).scatter_(dim = 0, index = torch.tensor(y), value = 1)
)


# data
def get_dataset(train_transform, test_transform, target_transform):
    train_dataset = datasets.FashionMNIST(
        root = "./data/",
        train = True,
        download = True,
        transform = train_transform,
        target_transform = target_transform,
    )
    test_dataset = datasets.FashionMNIST(
        root = "./data/",
        train = False,
        download = True,
        transform = test_transform,
        target_transform = target_transform,
    )
    
    return train_dataset, test_dataset


def get_dataloader(train_dataset, test_dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 4,
    )

    return train_loader, test_loader




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
