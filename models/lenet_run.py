# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lenet_run.py
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

from lenet.LeNet5 import LeNet5

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# deivce
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")


def train(model, train_loader, valid_loader, num_epochs, learning_rate):
    """
    model training
    """
    # loss
    loss_fn = nn.CrossEntropyLoss()

    ## optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # total step
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
            # error
            if (i+1) % 400 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}")


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
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')




# 测试代码 main 函数
def main():
    import torchvision.transforms as transforms

    from data_provider.MNIST import get_dataset, get_dataloader
    # params
    num_classes = 10
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001
    # ------------------------------
    # data
    # ------------------------------
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1307,), std = (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.1325,), std = (0.3105,))
    ])
    train_dataset, test_dataset = get_dataset(train_transform, test_transform)
    train_loader, test_loader = get_dataloader(train_dataset, test_dataset, batch_size = batch_size)
    # model
    model = LeNet5()
    print(model)
    

if __name__ == "__main__":
    main()
