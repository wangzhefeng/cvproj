# -*- coding: utf-8 -*-


# ***************************************************
# * File        : ResNet.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-23
# * Version     : 0.1.032308
# * Description : https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import numpy as np
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from cv_data.CIFAR10 import get_dataset, get_dataloader


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
num_classes = 10
num_epochs = 20
batch_size = 16
random_seed = 42
valid_size = 0.1
learning_rate = 0.01
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
class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x
        out += residual
        out = self.relu(out)
        return out

# TODO
class Residual(nn.Module):
    """
    The Residual block of ResNet.
    """
    def __init__(self, input_channels, num_channels, use_1x1conv = False, strides = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size = 3, padding = 1, stride = strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size = 3, padding = 1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size = 1, stride = strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes) -> None:
        super(ResNet, self).__init__()
        # TODO
        self.inplanes = 64
        # layers 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        # layers 2, 3, 4, 5
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        # average pooling
        self.avgpool = nn.AvgPool2d(7, stride = 1)
        # linear fc
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(planes),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


# model
model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes).to(device)
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
        
# ------------------------------
# model valid
# ------------------------------
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

# ------------------------------
# model testing
# ------------------------------
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
    pass

if __name__ == "__main__":
    main()
