# -*- coding: utf-8 -*-


# ***************************************************
# * File        : fnn.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-23
# * Version     : 0.1.032309
# * Description : Feedforward Neural Network with PyTorch
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
import torchvision
from torchvision import transforms


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
learning_rate = 0.1


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
class FNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(FNN, self).__init__()
        # linear
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 784 -> 100
        # non-linear
        self.sigmoid = nn.Sigmoid()
        # linear-readout
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 100 -> 10
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out


# ------------------------------
# model training
# ------------------------------
# model
input_dim = 28 * 28
hidden_dim = 100
output_dim = 10
model = FNN(input_dim, hidden_dim, output_dim)

# loss
loss_fn = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
