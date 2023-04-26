# -*- coding: utf-8 -*-


# ***************************************************
# * File        : AlexNet.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-23
# * Version     : 0.1.032305
# * Description : https://blog.paperspace.com/alexnet-pytorch/
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from cv_data.CIFAR10 import get_dataset, get_dataloader


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
num_classes = 10
batch_size = 64
num_epochs = 20
valid_size = 0.1
learning_rate = 0.005
random_seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Using device {device}.")


# ------------------------------
# data
# ------------------------------
# transforms
train_valid_normalize = transforms.Normalize(
    mean = [0.4914, 0.4822, 0.4465],
    std = [0.2023, 0.1994, 0.2010],
)
test_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
valid_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    train_valid_normalize,
])
train_transform_augment = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    train_valid_normalize,
])
train_transfrom = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    train_valid_normalize,
])
test_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    test_normalize,
])
# dataset
train_dataset, test_dataset, valid_dataset = get_dataset(
    train_transforms = train_transfrom,
    test_transforms = test_transform,
    valid_transforms = valid_transform,
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
# data loader
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
class AlexNet(nn.Module):

    def __init__(self, num_classes) -> None:
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 0),  # 3@227x227 -> 96@55x55
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),  # 96@55x55 -> 96@27x27
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2),  # 96@27x27 -> 256@27x27
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),  # 256@27x27  -> 256@13x13
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1),  # 256@13x13 -> 384@13x13
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1),  # 384@13x13 -> 384@13x13
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),  # 384@13x13 -> 256@13x13
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),  # 256@13x13 -> 256@6x6
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),  # 9216 -> 4096
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),  # 4096 -> 4096
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes),  # 4096 -> 10
        )
    
    def forward(self, x):
        """
        shape of x: 3@227x227
        """
        x = self.layer1(x)  # 3@227x227 -> 96@27x27
        x = self.layer2(x)  # 96@27x27 -> 256@13x13
        x = self.layer3(x)  # 256@13x13 -> 384@13x13
        x = self.layer3(x)  # 384@13x13 -> 384@13x13
        x = self.layer5(x)  # 384@13x13 -> 256@6x6
        x = x.reshape(x.size(0), -1)  # 256@6x6 -> 9216
        x = self.fc1(x)  # 9216 -> 4096
        x = self.fc2(x)  # 4096 -> 4096
        out = self.fc3(x)  # 4096 -> 10
        return out


# model
model = AlexNet(num_classes)
print(model)

# ------------------------------
# model training
# ------------------------------
# loss
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(
    model.parameters(), 
    lr = learning_rate, 
    weight_decay = 0.005, 
    momentum = 0.9
)

# modle train
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
    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}")
    # valid
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            # data
            images = images.to(device)
            labels = labels.to(device)
            # predict
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print(f"Accuracy of the network on the {5000} validation images: {100 * correct / total}")

# ------------------------------
# model testing
# ------------------------------
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # data
        images = images.to(device)
        labels = labels.to(device)
        # predict
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs

    print(f"Accuracy of the network on the {10000} test images: {100 * correct / total}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
