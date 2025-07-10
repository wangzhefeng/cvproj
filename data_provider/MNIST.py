# -*- coding: utf-8 -*-

# ***************************************************
# * File        : MNIST.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091417
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = [
    "get_dataloader"
]

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torchvision import datasets
import torchvision.transforms as transforms

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def get_dataset(train_transforms, test_transforms):
    """
    Dataset
    """
    train_dataset = datasets.MNIST(
        root = "./dataset/",
        train = True,
        transform = train_transforms,
        download = True,
    )
    test_dataset = datasets.MNIST(
        root = "./dataset/",
        train = False,
        transform = test_transforms,
        download = True,
    )

    return train_dataset, test_dataset


def get_dataloader(batch_size, num_workers = 0):
    """
    DataLoader
    """
    train_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1307,), std = (0.3081,))
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1325,), std = (0.3105,))
    ])
    # Dataset
    train_dataset, test_dataset = get_dataset(train_transforms, test_transforms)
    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
    )

    return train_loader, None, test_loader




# 测试代码 main 函数
def main():
    # params
    batch_size = 64
    # data
    train_loader, valid_loader, test_loader = get_dataloader(batch_size=batch_size)
    for batch in train_loader:
        break
    logger.info(f"image: \n{batch[0]} \nimage.shape{batch[0].shape}")
    logger.info(f"labels: \n{batch[1]} \nlabels.shape{batch[0].shape}")

if __name__ == "__main__":
    main()
