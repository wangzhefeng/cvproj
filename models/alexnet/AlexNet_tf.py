# -*- coding: utf-8 -*-


# ***************************************************
# * File        : AlexNet_tf.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-27
# * Version     : 0.1.032702
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import tensorflow as tf
from tensorflow import keras
from d2l import tensorflow as d2l


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def net():
    model = tf.keras.models.Sequential([
        tf.kears.layers.Conv2D(filters = 96, kernel_size = 11, strides = 4, activation = "relu"),
        tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2),
        tf.keras.layers.Conv2D(filters = 256, kernel_size = 5, padding = "same", activation = "relu"),
        tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2),
        tf.keras.layers.Conv2D(filters = 384, kernel_size = 3, padding = "same", activation = "relu"),
        tf.keras.layers.Conv2D(filters = 384, kernel_size = 3, padding = "same", activation = "relu"),
        tf.keras.layers.Conv2D(filters = 256, kernel_size = 3, padding = "same", activation = "relu"),
        tf.keras.layers.MaxPool2D(pool_size = 3, strides = 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation = "relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation = "relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10),
    ])

    return model


class AlexNet(tf.keras.models):
    pass






X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, "output shape: \t", X.shape)






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
