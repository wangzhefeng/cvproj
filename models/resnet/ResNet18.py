# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ResNet18.py
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
from torch import nn
import torchvision
# from d2l import torch as d2l
import utils.d2l_torch as d2l

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
devices = d2l.try_all_gpus()


# 预训练模型
pretrained_net = torchvision.models.resnet18(weights = "ResNet18_Weights.DEFAULT")
print(pretrained_net.fc)
# 微调模型
finetune_net = torchvision.models.resnet18(weights = "ResNet18_Weights.DEFAULT")
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
