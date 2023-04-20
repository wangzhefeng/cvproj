# -*- coding: utf-8 -*-


# ***************************************************
# * File        : MobileNet_v3_todo.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-29
# * Version     : 0.1.032911
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
from torchvision.models import (
    mobilenetv3,
    mobilenet_v3_small,
    mobilenet_v3_large,
    MobileNetV3,
    MobileNet_V3_Small_Weights,
    MobileNet_V3_Large_Weights,
    get_model_builder,
)


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


net = mobilenet_v3_small(
    weights = MobileNet_V3_Small_Weights.DEFAULT,
    progress = False,
)
net.eval()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
