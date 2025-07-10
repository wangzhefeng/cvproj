# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Classifier.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091417
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch import nn

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}.")


class Classifier(nn.Module):

    def __init__(self, features) -> None:
        super(Classifier, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(features, 5),
            nn.ReLU(),
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 15),
            nn.ReLU(),
        )
        self.final_layer = nn.Sequential(
            nn.Linear(15, 2),
            nn.Softmax(1)
        )
    
    def forward(self, x):
        x = self.first_layer(x)
        out = self.final_layer(x)
        
        return out




# 测试代码 main 函数
def main():
    net = Classifier(features = 2).to(device)
    print(net)

if __name__ == "__main__":
    main()
