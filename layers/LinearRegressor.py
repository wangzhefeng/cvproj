# -*- coding: utf-8 -*-


# ***************************************************
# * File        : LinearRegression.py
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

import numpy as np
import torch
import torch.nn as nn


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}.")
num_epochs = 100
learning_rate = 0.01


# ------------------------------
# data
# ------------------------------
# x
x_values = range(11)
x_train = np.array(x_values, dtype = np.float32)
x_train = x_train.reshape(-1, 1)
# y
y_values = [2 * i + 1 for i in x_values]
y_train = np.array(y_values, dtype = np.float32)
y_train = y_train.reshape(-1, 1)

print(x_train)
print(y_train)

# ------------------------------
# model
# ------------------------------
class LinearRegressor(nn.Module):

    def __init__(self, input_dim, output_dim) -> None:
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

# ------------------------------
# model training
# ------------------------------
# model
model = LinearRegressor(input_dim = 1, output_dim = 1).to(device)

# loss
loss_fn = nn.MSELoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# model training
for epoch in range(num_epochs):
    epoch += 1
    # numpy array -> torch Variable
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)
    # clear gradients w.r.t parameters
    optimizer.zero_grad()
    # forward
    outputs = model(inputs)
    # loss
    loss = loss_fn(outputs, labels)
    # get gradients w.r.t parameters
    loss.backward()
    # update parameters
    optimizer.step()

    print(f"epoch {epoch}, loss {loss.item()}")

# model predict
prediction = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
print(prediction)

# ------------------------------
# model save
# ------------------------------
save_model = True
if save_model:
    # only parameters
    torch.save(model.state_dict(), "./model/linear_regression.pkl")


# ------------------------------
# model load
# ------------------------------
load_model = False
if load_model:
    model.load_state_dict(torch.load("./model/linear_regression.pkl"))









# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
