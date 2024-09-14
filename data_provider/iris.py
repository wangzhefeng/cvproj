# -*- coding: utf-8 -*-

# ***************************************************
# * File        : IRIS.py
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

from sklearn import datasets
import torch
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split,
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def __get_dataset():
    """
    Dataset
    """
    # data
    iris_data = datasets.load_iris()
    # dataset 
    dataset_iris = TensorDataset(
        torch.tensor(iris_data.data),
        torch.tensor(iris_data.target),
    )
    # train and test dataset split
    num_train = int(len(dataset_iris) * 0.8)
    num_val = len(dataset_iris) - num_train
    train_dataset, test_dataset = random_split(
        dataset_iris,
        [num_train, num_val],
    )
    
    return train_dataset, test_dataset


def get_dataloader(batch_size = 8):
    """
    DataLoader
    """
    # Dataset
    train_dataset, test_dataset = __get_dataset()
    # DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 2,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 2,
    )
    
    return train_dataloader, test_dataloader




# 测试代码 main 函数
def main():
    train_dataloader, test_loader = get_dataloader(batch_size = 8)
    # test
    for features, labels in train_dataloader:
        print(features, labels)
        break

if __name__ == "__main__":
    main()
