# -*- coding: utf-8 -*-


# ***************************************************
# * File        : CNN.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-21
# * Version     : 0.1.032122
# * Description : CNN
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import os
import tensorflow as tf


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters = 32,           # 卷积层神经元(卷积核)的数目
            kernel_size = [5, 5],   # 感受野的大小
            padding = "same",       # padding 策略(vaild 或 same)
            activation = tf.nn.relu # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size = [2, 2], strides = 2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size = [5, 5],
            padding =  "same",
            activation = tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size = [2, 2], strides = 2)
        self.flatten = tf.keras.layers.Reshape(target_shape = (7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units = 1024, activation = tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units = 10)
    
    def call(self, inputs):    # [batch_size, 28, 28, 32]
        print(inputs.shape)
        x = self.conv1(inputs) # [batch_size, 28, 28, 32]
        print(x.shape)
        x = self.pool1(x)     # [batch_size, 14, 14, 32]
        print(x.shape)
        x = self.conv2(x)     # [batch_size, 14, 14, 64]
        print(x.shape)
        x = self.pool2(x)     # [batch_size, 7, 7, 64]
        print(x.shape)
        x = self.flatten(x)   # [batch_size, 7 * 7 * 64]
        print(x.shape)
        x = self.dense1(x)    # [batch_size, 1024]
        x = self.dense2(x)    # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


model = CNN()



def main():
    pass


if __name__ == "__main__":
    main()

