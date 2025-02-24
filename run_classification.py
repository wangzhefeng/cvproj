# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run_classification.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-10-20
# * Version     : 0.1.102003
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import argparse

import gc
import torch
import torch.nn as nn

from utils import device

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]

# TODO params
batch_size = 100
learning_rate = 0.1
n_iters = 3000
train_dataset = 10000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))
num_epochs = 100


def train(model, train_loader, valid_loader, num_epochs, learning_rate):
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
        momentum = 0.9,
    )
    # TODO
    # optimizer = torch.optim.Adam(
    #     model.parameters(), 
    #     lr = learning_rate,
    # )
    # model train(run epochs)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        # run batches
        for i, (images, labels) in enumerate(train_loader):
            # data
            images = images.to(device)
            labels = labels.to(device)
            # TODO clear gradient
            optimizer.zero_grad()
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

            if (i+1) % 400 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch: {i}, Step [{i+1}/{total_step}], Loss: {loss.item()}")
        print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item()}")

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
            # predict(forward)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # accuracy(loss)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # TODO del images, labels, outputs
            print(f"Batch: {i}, Accuracy of the network on {len(labels)}: {correct / total}")
        print(f"Accuracy of the network on the {test_loader.size()} test images: {100 * correct / total}")


def train_fine_tuning(net, train_iter, test_iter, learning_rate, num_epochs = 5, param_group = True):
    # from d2l import torch as d2l
    import utils.d2l_torch as d2l
    devices = d2l.try_all_gpus()

    # loss
    loss = nn.CrossEntropyLoss(reduction = "none")
    # train
    if param_group:
        params_1x = [
            param 
            for name, param in net.named_parameters() 
            if name not in ["fc.weight", "fc.bias"]
        ]
        trainer = torch.optim.SGD(
            params = [
                {"params": params_1x},
                {"params": net.fc.parameters(), "lr": learning_rate * 10}
            ],
            lr = learning_rate,
            weight_decay = 0.001,
        )
    else:
        trainer = torch.optim.SGD(
            params = net.parameters(), 
            lr = learning_rate, 
            weight_decay = 0.001
        )

    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


def model_save_load(model):
    # model save
    save_model = True
    if save_model:
        # only parameters
        torch.save(model.state_dict(), "./saved_model/linear_regression.pkl")

    # model load
    load_model = False
    if load_model:
        model.load_state_dict(torch.load("./saved_model/linear_regression.pkl"))




# 测试代码 main 函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument()

if __name__ == "__main__":
    main()
