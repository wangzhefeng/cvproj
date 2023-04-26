# -*- coding: utf-8 -*-


# ***************************************************
# * File        : inceptionv3.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-25
# * Version     : 0.1.032518
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
# 禁用所有与模型训练有关的操作
keras.backend.set_learning_phase(0)


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# file dir
base_dir = os.path.dirname(__file__)
print(base_dir)
model_dir = os.path.join(base_dir, "models")


# ------------------------------
# model
# ------------------------------
# model
model = keras.applications.inception_v3.InceptionV3(
    weights = "imagenet",  # 使用预训练的 ImageNet 权重加载模型 
    include_top = False,  # 不包括全连接层
)
# model plot
keras.utils.plot_model(
    model, 
    os.path.join(model_dir, "inception_v3.png"), 
    show_shapes = True
)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
