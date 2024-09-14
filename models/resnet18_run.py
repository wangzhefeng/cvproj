# -*- coding: utf-8 -*-

# ***************************************************
# * File        : resnet18_run.py
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
from torchvision import datasets, transforms
# from d2l import torch as d2l
import utils.d2l_torch as d2l

from resnet.ResNet18 import finetune_net

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# deivce
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
devices = d2l.try_all_gpus()
print(f"Using device {device}.")


def train_fine_tuning(net, train_iter, test_iter, learning_rate, num_epochs = 5, param_group = True):
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
    


# 测试代码 main 函数
def main():
    # params
    batch_size = 128
    # ------------------------------
    # data
    """
    hotdog
        - train
            - hotdog
            - not-hotdog
        - test
            - hotdog
            - not-hotdog
    """
    # ------------------------------
    # data download
    d2l.DATA_HUB["hotdog"] = (
        d2l.DATA_URL + "hotdog.zip", 
        "fba480ffa8aa7e0febbb511d181409f899b9baa5"
    )
    data_dir = d2l.download_extract("hotdog")
    print(data_dir)

    # transforms
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],  # RGB 通道的均值
        std = [0.229, 0.224, 0.225],  # RGB 通道的标准差
    )
    train_augs = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_augs = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # dataset
    train_imgs = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform = train_augs,
    )
    test_imgs = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform = test_augs,
    )

    # dataloader
    train_iter = torch.utils.data.DataLoader(
        train_imgs,
        batch_size = batch_size,
        shuffle = True,
    )
    test_iter = torch.utils.data.Dataloader(
        test_imgs,
        batch_size = batch_size,
        shuffle = False,
    )
    # ------------------------------
    # 
    # ------------------------------
    train_fine_tuning(finetune_net, train_iter, test_iter, learning_rate = 5e-5)

if __name__ == "__main__":
    main()
