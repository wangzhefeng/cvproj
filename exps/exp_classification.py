# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_classification.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-24
# * Version     : 1.0.022416
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from data_provider.MNIST import get_dataloader
from exps.exp_basic import Exp_Basic
from models.lenet import LeNet5
from models.alexnet import AlexNet
from utils.argsparser_tools import DotDict
from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


model_dict = {
    "lenet5": LeNet5,
    "alxenet": AlexNet,
}


class Exp_Classification(Exp_Basic):
    
    def __init__(self, args):
        super().__init__(args)

    def _build_data(self):
        """
        build data
        """
        train_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.1307,), std = (0.3081,))
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.1325,), std = (0.3105,))
        ])
        self.train_loader, self.test_loader = get_dataloader(
            batch_size=self.args.batch_size, 
            train_transforms = train_transforms, 
            test_transforms=test_transforms, 
            num_workers = 0
        )
    
    def _build_model(self):
        """
        build model
        """
        # model instance
        model = model_dict[self.args.model_name].Model(self.args).float()
        # 单机多卡训练
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids = self.args.device_ids)
        # print model params nums
        total = sum([param.nelement() for param in model.parameters()])
        logger.info(f"Number of parameters: {(total / 1e6):.2f}M")

        return model

    def _select_optimizer(self):
        """
        optimizer
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr = self.args.learning_rate, 
            weight_decay = self.args.weight_decay
        )

        return optimizer

    def _select_criterion(self):
        """
        loss
        """
        criterion = nn.CrossEntropyLoss()

        return criterion

    def train(self):
        # data
        self._build_data()
        # loss
        criterion = self._select_criterion()
        # optimizer
        optimizer = self._select_optimizer()
        # how many steps are remaining when trainign
        total_step = len(self.train_loader)
        for epoch in range(self.args.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                # forward
                outputs = self.model(images)
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                # backward
                loss.backward()
                optimizer.step()
                if (i + 1) % 400 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.args.num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")

    def valid(self):
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            logger.info(f"Accuracy of the network on the 10000 test images: {100 * correct / total} %")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
