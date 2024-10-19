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
# * TODO        : 1.
# ***************************************************

__all__ = [
    "transforms_cifar",
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


# data preprocessing
transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    )
])


def get_dataset(train_transforms, test_transforms, valid_transforms = None):
    """
    Dataset
    """
    train_dataset = datasets.CIFAR10(
        root = "./data/",
        train = True,
        download = True,
        transform = train_transforms,
    )
    test_dataset = datasets.CIFAR10(
        root = "./data/",
        train = False,
        download = True,
        transform = test_transforms,
    )
    if valid_transforms:
        valid_dataset = datasets.CIFAR10(
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
        train_dataset, test_dataset = get_dataset(
            train_transforms, 
            test_transforms
        )
    else:
        train_dataset, test_dataset, valid_dataset = get_dataset(
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
    import numpy as np
    from torchvision import transforms
    from torch.utils.data.sampler import SubsetRandomSampler
    # ------------------------------
    # params
    # ------------------------------
    num_classes = 10
    batch_size = 64
    num_epochs = 20
    valid_size = 0.1
    learning_rate = 0.005
    random_seed = 42
    # ------------------------------
    # data
    # ------------------------------
    # transforms
    train_transform_augment = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.2023, 0.1994, 0.2010],
        ),
    ])
    train_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.2023, 0.1994, 0.2010],
        ),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225],
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
    # ------------------------------
    # data split
    # ------------------------------ 
    train_dataset, _, _ = get_dataset(
        train_transform,
        test_transform, 
        valid_transform
    )
    num_train = len(train_dataset)
    num_valid = int(np.floor(valid_size * num_train))

    indices = list(range(num_train))    
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[num_valid:], indices[:num_valid]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # data loader
    train_loader, test_loader, valid_loader = get_dataloader(
        batch_size = batch_size,
        train_transforms = train_transform,
        test_transforms = test_transform,
        valid_transforms = valid_transform,
        train_sampler = train_sampler,
        valid_sampler = valid_sampler,
        num_workers = 1,
    )

if __name__ == "__main__":
    main()
