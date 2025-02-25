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

import torch.nn as nn

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model(nn.Module):

    def __init__(self, args) -> None:
        super(Model, self).__init__()
        
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
            nn.Linear(4096, args.num_classes),  # 4096 -> 10
        )
    
    def forward(self, x):
        """
        shape of x: 3@227x227
        """
        x = self.layer1(x)  # 3@227x227 -> 96@27x27
        x = self.layer2(x)  # 96@27x27 -> 256@13x13
        x = self.layer3(x)  # 256@13x13 -> 384@13x13
        x = self.layer4(x)  # 384@13x13 -> 384@13x13
        x = self.layer5(x)  # 384@13x13 -> 256@6x6
        x = x.reshape(x.size(0), -1)  # 256@6x6 -> 9216
        x = self.fc1(x)  # 9216 -> 4096
        x = self.fc2(x)  # 4096 -> 4096
        out = self.fc3(x)  # 4096 -> 10
        return out




# 测试代码 main 函数
def main():
    from utils.argsparser_tools import DotDict
    from utils.log_util import logger
    # args
    args = {
        "num_classes": 10,
        "batch_size": 64,
        "num_epochs": 20,
        "learning_rate": 0.005,
        "random_seed": 42,
    }
    args = DotDict(args)
    # model
    model = Model(args)
    logger.info(f"model: \n{model}")

if __name__ == "__main__":
    main()
