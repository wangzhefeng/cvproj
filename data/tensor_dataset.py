# -*- coding: utf-8 -*-


# ***************************************************
# * File        : dataset.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-26
# * Version     : 0.1.032622
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
from sklearn import datasets
import torch
from torch.utils.data import (
    TensorDataset,
    Dataset,
    DataLoader,
    random_split,
)


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


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
train_dataset, valid_dataset = random_split(
    dataset_iris,
    [num_train, num_val],
)

# dataloader
train_dataloader = DataLoader(
    train_dataset,
    batch_size = 8,
    shuffle = True,
)
test_dataloader = DataLoader(
    valid_dataset,
    batch_size = 8,
    shuffle = False,
)

# test
for features, labels in train_dataloader:
    print(features, labels)
    break

# 演示加法运算符 `+` 的合并作用
dataset_iris = train_dataset + valid_dataset
print(f"len(train_dataset) = {len(train_dataset)}")
print(f"len(valid_dataset) = {len(valid_dataset)}")
print(f"len(train_dataset+valid_dataset) = {len(dataset_iris)}")
print((type(dataset_iris)))




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
