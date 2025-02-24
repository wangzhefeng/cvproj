# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LeNet5.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-23
# * Version     : 0.1.032304
# * Description : https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/
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

from utils.log_util import logger

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()

        # layer 1
        self.layer1 = nn.Sequential(
            # conv 6@5x5
            nn.Conv2d(in_channels = args.in_channels, out_channels = 6, kernel_size = 5, stride = 1, padding = 0),  # 1@32x32 -> 6@28x28
            nn.BatchNorm2d(num_features = 6),
            nn.ReLU(),
            # pooling 2x2
            nn.MaxPool2d(kernel_size = 2, stride = 2),  # 6@28x28 -> 6@14x14
        )
        # layer 2
        self.layer2 = nn.Sequential(
            # conv 16@5x5
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1, padding = 0),  # 6@14x14 -> 16@10x10
            nn.BatchNorm2d(num_features = 16),
            nn.ReLU(),
            # pooling 2x2
            nn.MaxPool2d(kernel_size = 2, stride = 2),  # 16@10x10 -> 16@5x5
        )
        # layer 3 
        # conv filter: 120@5x5
        self.fc1 = nn.Linear(in_features = 400, out_features = 120)  # 16@5x5 -> 120
        self.relu = nn.ReLU()
        # layer 4
        self.fc2 = nn.Linear(in_features = 120, out_features = 84)  # 120 -> 84
        self.relu1 = nn.ReLU()
        # layer 5
        self.fc3 = nn.Linear(in_features = 84, out_features = args.num_classes)  # 84 -> 10
    
    def forward(self, x):
        """
        shape of x: 1@32x32
        """
        x = self.layer1(x)  # 1@32x32 -> 6@14x14
        x = self.layer2(x)  # 6@14x14 -> 16@5x5
        x = x.reshape(x.size(0), -1)  # 16@5x5 -> 400
        x = self.fc1(x)  # 400 -> 120
        x = self.relu(x)
        x = self.fc2(x)  # 120 -> 84
        x = self.relu1(x)
        out = self.fc3(x)  # 84 -> 10
        
        return out




# 测试代码 main 函数
def main():
    from utils.argsparser_tools import DotDict
    # args
    args = {
        "in_channels": 1,
        "num_classes": 10,
    }
    args = DotDict(args)
    # model
    model = Model(args)
    logger.info(model)

if __name__ == "__main__":
    main()
