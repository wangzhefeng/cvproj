# -*- coding: utf-8 -*-


# ***************************************************
# * File        : model_load.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-20
# * Version     : 0.1.032000
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import tensorflow as tf
from tensorflow import keras
from config.config_loader import settings


def model_load():
    """
    模型加载
    """
    model = tf.saved_model.load("saved/1")
    model = tf.keras.models.load_model(settings["PATH"]["model_path"])
    
    model.summary()
    
    return model




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()



