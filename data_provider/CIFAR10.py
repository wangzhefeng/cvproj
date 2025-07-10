# -*- coding: utf-8 -*-

# ***************************************************
# * File        : CIFAR10.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-15
# * Version     : 1.0.091500
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "transforms_cifar",
    "get_dataloader",
]

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# data preprocessing
transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    )
])


def get_dataset(train, transforms):
    """
    Dataset
    """
    dataset = datasets.CIFAR10(
        root = "./dataset/",
        train = train,
        download = True,
        transform = transforms,
    )
    
    return dataset


def get_train_valid_loader(batch_size,
                           argument = False,
                           shuffle = True,
                           num_workers = 0):
    """
    DataLoader
    """
    # transforms
    if argument:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.4914, 0.4822, 0.4465],
                std = [0.2023, 0.1994, 0.2010],
            ),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.4914, 0.4822, 0.4465],
                std = [0.2023, 0.1994, 0.2010],
            ),
        ])
    valid_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.2023, 0.1994, 0.2010],
        ),
    ])
    # Dataset
    train_dataset = get_dataset(train = True, transforms = train_transform)
    valid_dataset = get_dataset(train = True, transforms = valid_transform)
    # data sampler
    num_train = len(train_dataset)
    num_valid = int(np.floor(0.1 * num_train)) 
    indices = list(range(num_train))
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[num_valid:], indices[:num_valid]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        # shuffle = shuffle,
        sampler = train_sampler,
        num_workers = num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = batch_size,
        # shuffle = shuffle,
        sampler = valid_sampler,
        num_workers = num_workers,
    )
    
    return train_loader, valid_loader


def get_test_loader(batch_size, 
                    shuffle = False,
                    num_workers = 0):
    """
    DataLoader
    """
    # transforms
    test_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225],
        ),
    ])
    # Dataset
    test_dataset = get_dataset(train = False, transforms = test_transform)
    # DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
    )
    
    return test_loader


def get_dataloader(batch_size, num_workers = 0):
    train_loader, valid_loader = get_train_valid_loader(
        batch_size = batch_size,
        argument=False,
        shuffle = True,
        num_workers = num_workers,
    )
    test_loader = get_test_loader(
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
    )
    
    return train_loader, valid_loader, test_loader




# 测试代码 main 函数
def main():
    # params
    use_valid = True
    batch_size = 64
    # data
    train_loader, valid_loader, test_loader = get_dataloader(batch_size = batch_size)
    # test
    for images, labels in train_loader:
        break
    logger.info(f"images: \n{images} \nimages.shape{images.shape}")
    logger.info(f"labels: \n{labels} \nlabels.shape{labels.shape}")

if __name__ == "__main__":
    main()
