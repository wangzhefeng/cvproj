# -*- coding: utf-8 -*-

# ***************************************************
# * File        : hotdog.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-27
# * Version     : 0.1.032717
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os

import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from d2l import torch as d2l

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
devices = d2l.try_all_gpus()


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
# model
# ------------------------------
# 预训练模型
pretrained_net = torchvision.models.resnet18(weights = "ResNet18_Weights.DEFAULT")
print(pretrained_net.fc)
# 微调模型
finetune_net = torchvision.models.resnet18(weights = "ResNet18_Weights.DEFAULT")
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)


def train_fine_tuning(net, learning_rate, num_epochs = 5, param_group = True):
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
    
train_fine_tuning(net = finetune_net, learning_rate = 5e-5)







# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
