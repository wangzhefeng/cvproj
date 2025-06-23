# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-22
# * Version     : 1.0.062220
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import sampler
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def get_dataloaders_cifar10(batch_size, 
                            num_workers=0,
                            validation_fraction=None,
                            train_transforms=None,
                            test_transforms=None,
                            download=True):
    # transforms
    if train_transforms is None:
        train_transforms = transforms.ToTensor()
    if test_transforms is None:
        test_transforms = transforms.ToTensor()
    # dataset
    train_dataset = datasets.CIFAR10(
        root='./dataset',
        train=True,
        transform=train_transforms,
        download=download
    )
    valid_dataset = datasets.CIFAR10(
        root='./dataset',
        train=True,
        transform=test_transforms
    )
    test_dataset = datasets.CIFAR10(
        root='./dataset',
        train=False,
        transform=test_transforms
    )
    # dataloader
    if validation_fraction is not None:
        num = int(validation_fraction * 50000)
        train_indices = range(0, 50000 - num)
        valid_indices = range(50000 - num, 50000)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)
 
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            sampler=valid_sampler,
            drop_last=False,
            num_workers=num_workers,
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    # return dataloader
    if validation_fraction is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
