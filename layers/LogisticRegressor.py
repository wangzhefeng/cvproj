# -*- coding: utf-8 -*-


# ***************************************************
# * File        : LogisticRegression.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-23
# * Version     : 0.1.032310
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision import transforms


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
learning_rate = 0.001


# ------------------------------
# data
# ------------------------------
train_dataset = torchvision.datasets.MNIST(
    root = "./data",
    train = True,
    transform = transforms.Compose([
        transforms.ToTensor(),
    ]),
    download = True,
)

test_dataset = torchvision.datasets.MNIST(
    root = "./data",
    train = False,
    transform = transforms.Compose([
        transforms.ToTensor(),
    ]),
    download = True,
)

# ------------------------------
# hyperparameters
# ------------------------------
n_iters = 3000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))

# ------------------------------
# data pipeline
# ------------------------------
train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,
)

test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False,
)

# ------------------------------
# model
# ------------------------------
class LogisticRegressor(nn.Module):

    def __init__(self, input_dim, output_dim) -> None:
        super(LogisticRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

# ------------------------------
# model training
# ------------------------------
# model
input_dim = 28 * 28
output_dim = 10
model = LogisticRegressor(input_dim, output_dim).to(device)

# loss
loss_fn = nn.CrossEntropyLoss()

# optimier
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# model training
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # load images as Variable
        images = images.view(-1, 28 * 28).requires_grad_().to(device)
        labels = labels
        # clear gradient
        optimizer.zero_grad()
        # forward
        outputs = model(images)
        # loss
        loss = loss_fn(outputs, labels)
        # get gradient
        loss.backward()
        # update parameters
        optimizer.step()

        iter += 1
        if iter % 500 == 0:
            # accuracy
            correct = 0
            total = 0
            # iterate test dataset
            for images, labels in test_loader:
                images = images.view(-1, 28 * 28).to(device)
                # forward
                outputs = model(images)
                # predict
                _, predicted = torch.max(outputs.data, 1)
                # total number of labels
                total += labels.size(0)
                # total correct predictions
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()
            accuracy = 100 * correct.item() / total
            # print loss
            print(f"Iteration: {iter}. Loss: {loss.item()}. Accuracy: {accuracy}")


# ------------------------------
# model save
# ------------------------------
save_model = False
if save_model:
    torch.save(model.state_dict(), "logistic_regression.pkl")





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
