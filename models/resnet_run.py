# -*- coding: utf-8 -*-

# ***************************************************
# * File        : resnet_run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-15
# * Version     : 1.0.091504
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

import numpy as np
import gc

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from data_provider.CIFAR10 import get_dataset, get_dataloader
from resnet.ResNet import ResNet, ResidualBlock

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# deivce
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")


def train(model, train_loader, num_epochs, learning_rate):
    """
    model training
    """
    # loss
    loss_fn = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr = learning_rate, 
        weight_decay = 0.001, 
        momentum = 0.9
    )
    # model train
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # data 
            images = images.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 内存回收
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()
        if (i + 1) / 400 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step: {i+1}/{total_step}, Loss: {loss.item()}")


def valid(model, valid_loader): 
    """
    model valid
    """
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(valid_loader):
            # data
            images = images.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Accuracy of the network of the {50000} validation images: {correct / total}")


def test(model, test_loader):
    """
    model testing
    """
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            # data
            images = images.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Accuracy of the network of the {50000} test images: {correct / total}")



# 测试代码 main 函数
def main():
    # ------------------------------
    # params
    # ------------------------------
    num_classes = 10
    num_epochs = 20
    batch_size = 16
    random_seed = 42
    valid_size = 0.1
    learning_rate = 0.01
    # ------------------------------
    # data
    # ------------------------------
    normalize = transforms.Normalize(
        mean = [0.4914, 0.4822, 0.4465],
        std = [0.2023, 0.1994, 0.2010],
    )
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset, test_dataset, valid_dataset = get_dataset(
        train_transforms = transform,
        test_transforms = transform,
        valid_transforms = transform,
    )
    # data split
    num_train = len(train_dataset)
    indices = list(range(num_train))
    num_valid = int(np.floor(valid_size * num_train))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[num_valid:], indices[:num_valid]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader, test_loader, valid_loader = get_dataloader(
        train_dataset = train_dataset,
        test_dataset = test_dataset,
        batch_size = batch_size,
        train_sampler = train_sampler,
        valid_dataset = valid_dataset,
        valid_sampler = valid_sampler,
    )
    # ------------------------------
    # model
    # ------------------------------
    model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes).to(device)
    print(model)

if __name__ == "__main__":
    main()
