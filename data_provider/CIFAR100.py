# -*- coding: utf-8 -*-

# ***************************************************
# * File        : CIFAR100.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091419
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
from torchvision import datasets, transforms

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# data preprocessing
transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    )
])


def __get_dataset(train_transforms, test_transforms, valid_transforms = None):
    """
    Dataset
    """
    train_dataset = datasets.CIFAR100(
        root = "./data/",
        train = True,
        download = True,
        transform = train_transforms,
    )
    test_dataset = datasets.CIFAR100(
        root = "./data/",
        train = False,
        download = True,
        transform = test_transforms,
    )
    if valid_transforms:
        valid_dataset = datasets.CIFAR100(
            root = "./data/",
            train = True,
            download = True,
            transform = valid_transforms,
        )
        return train_dataset, test_dataset, valid_dataset
    else:
        return train_dataset, test_dataset


def get_dataloader(batch_size, 
                   train_transforms, 
                   test_transforms, 
                   valid_transforms = None,
                   train_sampler = None,
                   valid_sampler = None,
                   num_workers = -1):
    """
    DataLoader
    """
    # Dataset
    if valid_transforms is None:
        train_dataset, test_dataset = __get_dataset(
            train_transforms, 
            test_transforms
        )
    else:
        train_dataset, test_dataset, valid_dataset = __get_dataset(
            train_transforms, 
            test_transforms, 
            valid_transforms
        )
    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        sampler = train_sampler,
        num_workers = num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
    )
    if valid_transforms:
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size = batch_size,
            shuffle = True,
            sampler = valid_sampler,
            num_workers = num_workers,
        )
        return train_loader, test_loader, valid_loader
    else:
        return train_loader, test_loader




# 测试代码 main 函数
def main():
    # params
    batch_size = 128
    # DataLoader
    train_loader, test_loader = get_dataloader(
        batch_size = batch_size,
        train_transforms = transforms_cifar,
        test_transforms = transforms_cifar,
        num_workers = 1,
    )

if __name__ == "__main__":
    main()
