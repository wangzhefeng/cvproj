# -*- coding: utf-8 -*-


# ***************************************************
# * File        : VGG16.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-23
# * Version     : 0.1.032307
# * Description : https://blog.paperspace.com/vgg-from-scratch-pytorch/
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

from cv_data.CIFAR100 import get_dataset, get_dataloader


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
num_classes = 100
num_epochs = 20
batch_size = 64 
valid_size = 0.1
learning_rate = 0.005
random_seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")


# ------------------------------
# data
# ------------------------------
normalize = transforms.Normalize(
    mean = [0.4914, 0.4822, 0.4465],
    std = [0.2023, 0.1994, 0.2010],
)
transform = transforms.Compose([
    transforms.Resize((227, 227)),
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
class VGG16(nn.Module):

    def __init__(self, num_classes) -> None:
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out


# model
model = VGG16(num_classes)
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
    momentum = 0.9,
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
    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}")

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
