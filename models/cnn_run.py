# -*- coding: utf-8 -*-

# ***************************************************
# * File        : cnn_run.py
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

import torch
import torch.nn as nn

from cnn.CNN import CNN

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
        weight_decay = 0.005, 
        momentum = 0.9
    )
    # run epochs
    for epoch in range(num_epochs):
        # run batches
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
            print(f"Epoch [{epoch + 1} / {num_epochs}], Batch: {i}, Loss: {loss.item()}")
        print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item()}")


def test(model, train_loader):
    """
    model testing
    """
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            # data
            images = images.to(device)
            labels = labels.to(device)
            # forward
            outputs = model(images)
            # predict
            _, predicted = torch.max(outputs.data, 1)
            # loss
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f"Batch: {i}, Accuracy of the network on {len(labels)}: {correct / total}")
        print(f"Accuracy of the network on the {50000} train images: {correct / total}")


# 测试代码 main 函数
def main():
    from torchvision import transforms
    from data_provider.CIFAR10 import get_dataset, get_dataloader
    # ------------------------------
    # params
    # ------------------------------
    batch_size = 64
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 20
    # ------------------------------
    # data
    # ------------------------------
    all_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.2023, 0.1994, 0.2010],
        )
    ])
    train_dataset, test_dataset = get_dataset(
        train_transforms = all_transforms,
        test_transforms = all_transforms
    )
    train_loader, test_loader = get_dataloader(
        train_dataset = train_dataset,
        test_dataset = test_dataset,
        batch_size = batch_size
    )
    # ------------------------------
    # model
    # ------------------------------
    model = CNN(num_classes)
    print(model)

if __name__ == "__main__":
    main()
