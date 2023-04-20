# -*- coding: utf-8 -*-


# ***************************************************
# * File        : Classification.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-03
# * Version     : 0.1.040318
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import torch
from torch import nn


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Classifier(nn.Module):

    def __init__(self, features) -> None:
        super(Classifier, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(features, 5),
            nn.ReLU(),
        )
        self.second_layer = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
        )
        self.third_layer = nn.Sequential(
            nn.Linear(10, 15),
            nn.ReLU(),
        )
        self.final_layer = nn.Sequential(
            nn.Linear(15, 2),
            nn.Softmax(1)
        )
    
    def forward(self, x):
        out = self.first_layer(x)
        out = self.second_layer(out)
        out = self.third_layer(out)
        out = self.final_layer(out)
        return out


net = Classifier().to(device)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
