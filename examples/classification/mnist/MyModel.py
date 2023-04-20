# -*- coding: utf-8 -*-


# ***************************************************
# * File        : mnist_tf2_experts.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-21
# * Version     : 0.1.032122
# * Description : Model Template
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import os
import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation = "relu")
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation = "relu")
        self.d2 = tf.keras.layers.Dense(10, activation = "softmax")

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


model = MyModel()




def main():
    pass


if __name__ == "__main__":
    main()

