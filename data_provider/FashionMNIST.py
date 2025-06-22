# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FashionMNIST.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091417
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = [
    "train_transform",
    "test_transform",
    "target_transform",
    "get_dataloader",
]

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


# transform
train_transform = transforms.ToTensor()
test_transform = transforms.ToTensor()
target_transform = transforms.Lambda(
    lambda y: torch.zeros(10, dtype = torch.float)
                   .scatter_(dim = 0, index = torch.tensor(y), value = 1)
)


def __get_dataset(train_transform, test_transform, target_transform):
    """
    Dataset
    """
    train_dataset = datasets.FashionMNIST(
        root = "./dataset/",
        train = True,
        download = True,
        transform = train_transform,
        target_transform = target_transform,
    )
    test_dataset = datasets.FashionMNIST(
        root = "./dataset/",
        train = False,
        download = True,
        transform = test_transform,
        target_transform = target_transform,
    )
    
    return train_dataset, test_dataset


def get_dataloader(batch_size, train_transform, test_transform, target_transform):
    """
    DataLoader
    """
    train_dataset, test_dataset = __get_dataset(
        train_transform, 
        test_transform, 
        target_transform
    )
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
    # params
    batch_size = 128
    # DataLoader
    train_loader, test_loader = get_dataloader(
        batch_size = batch_size,
        train_transforms = train_transform,
        test_transforms = test_transform,
        num_workers = 1,
    )

if __name__ == "__main__":
    main()
