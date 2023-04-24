# -*- coding: utf-8 -*-

# ***************************************************
# * File        : train_val_loop.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-24
# * Version     : 0.1.042422
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import torch
from torch import nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = None
epochs = None


def train(dataloader, model, loss_fn, optimizer):
    """
    In a single training loop, the model makes predictions 
    on the training dataset (fed to it in batches), 
    and backpropagates the prediction error to adjust 
    the model’s parameters.
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # 计算预测误差
        pred = model(X)
        loss = loss_fn(pred, y)
        # 后向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    """
    check the model’s performance against 
    the test dataset to ensure it is learning
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # 将数据移动到设备上
            X, y = X.to(device), y.to(device)
            # 计算累计测试误差
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



train_dataloader = None
test_dataloader = None

model = None

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化算法
optimizer =  torch.optim.SGD(model.parameters(), lr = learning_rate)

for t in range(epochs):
    print(f"Epoch {t + 1} ---------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
