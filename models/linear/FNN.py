# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FNN.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091417
# * Description : Feedforward Neural Network with PyTorch
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

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}.")


class FNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(FNN, self).__init__()
        # linear
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 784 -> 100
        # non-linear
        self.sigmoid = nn.Sigmoid()
        # linear-readout
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 100 -> 10
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        
        return out




# 测试代码 main 函数
def main():
    # model
    input_dim = 28 * 28
    hidden_dim = 100
    output_dim = 10
    net = FNN(input_dim, hidden_dim, output_dim)

if __name__ == "__main__":
    main()
