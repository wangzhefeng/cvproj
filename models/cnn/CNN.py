# -*- coding: utf-8 -*-


# ***************************************************
# * File        : CNN.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-24
# * Version     : 0.1.032403
# * Description : https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
_path = os.path.abspath(os.path.dirname(__file__))
if os.path.join(_path, "..") not in sys.path:
    sys.path.append(os.path.join(_path, ".."))

import torch
import torch.nn as nn
from torchvision import transforms

from cv_data.CIFAR10 import get_dataset, get_dataloader


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
class CNN(nn.Module):

    def __init__(self, num_classes) -> None:
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride = 2)

        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)

        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


model = CNN(num_classes)
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



# ------------------------------
# model testing
# ------------------------------
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
    pass

if __name__ == "__main__":
    main()
