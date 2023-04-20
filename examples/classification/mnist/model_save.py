# -*- coding: utf-8 -*-


# ***************************************************
# * File        : model_save.py
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


def model_save(trained_model):
    """
    模型保存

    :param trained_model: _description_
    :type trained_model: _type_
    """
    tf.saved_model.save(trained_model, "save/1")
    trained_model.save(settings["PATH"]["model_path"])
    
    del model



# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()


