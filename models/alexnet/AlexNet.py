# -*- coding: utf-8 -*-

# ***************************************************
# * File        : AlexNet.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-23
# * Version     : 0.1.032305
# * Description : https://blog.paperspace.com/alexnet-pytorch/
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Using device {device}.")


class AlexNet(nn.Module):

    def __init__(self, num_classes) -> None:
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 0),  # 3@227x227 -> 96@55x55
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),  # 96@55x55 -> 96@27x27
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2),  # 96@27x27 -> 256@27x27
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),  # 256@27x27  -> 256@13x13
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1),  # 256@13x13 -> 384@13x13
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1),  # 384@13x13 -> 384@13x13
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),  # 384@13x13 -> 256@13x13
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),  # 256@13x13 -> 256@6x6
        )
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),  # 9216 -> 4096
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),  # 4096 -> 4096
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes),  # 4096 -> 10
        )
    
    def forward(self, x):
        """
        shape of x: 3@227x227
        """
        x = self.layer1(x)  # 3@227x227 -> 96@27x27
        x = self.layer2(x)  # 96@27x27 -> 256@13x13
        x = self.layer3(x)  # 256@13x13 -> 384@13x13
        x = self.layer3(x)  # 384@13x13 -> 384@13x13
        x = self.layer5(x)  # 384@13x13 -> 256@6x6
        x = x.reshape(x.size(0), -1)  # 256@6x6 -> 9216
        x = self.fc1(x)  # 9216 -> 4096
        x = self.fc2(x)  # 4096 -> 4096
        out = self.fc3(x)  # 4096 -> 10
        return out




# 测试代码 main 函数
def main():
    import numpy as np
    from torchvision import transforms
    from torch.utils.data.sampler import SubsetRandomSampler
    
    from data_provider.CIFAR10 import get_dataset, get_dataloader
    # ------------------------------
    # params
    # ------------------------------
    num_classes = 10
    batch_size = 64
    num_epochs = 20
    valid_size = 0.1
    learning_rate = 0.005
    random_seed = 42
    # ------------------------------
    # data
    # ------------------------------
    # transforms
    train_transform_augment = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.2023, 0.1994, 0.2010],
        ),
    ])
    train_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.2023, 0.1994, 0.2010],
        ),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225],
        ),
    ])
    valid_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.4914, 0.4822, 0.4465],
            std = [0.2023, 0.1994, 0.2010],
        ),
    ])
    # ------------------------------
    # data split
    # ------------------------------ 
    train_dataset, _, _ = get_dataset(
        train_transform,
        test_transform, 
        valid_transform
    )
    num_train = len(train_dataset)
    num_valid = int(np.floor(valid_size * num_train))

    indices = list(range(num_train))    
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[num_valid:], indices[:num_valid]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # data loader
    train_loader, test_loader, valid_loader = get_dataloader(
        batch_size = batch_size,
        train_transforms = train_transform,
        test_transforms = test_transform,
        valid_transforms = valid_transform,
        train_sampler = train_sampler,
        valid_sampler = valid_sampler,
        num_workers = 1,
    )
    # ------------------------------
    # model
    # ------------------------------
    model = AlexNet(num_classes)
    print(model)

if __name__ == "__main__":
    main()
