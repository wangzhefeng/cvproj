# -*- coding: utf-8 -*-


# ***************************************************
# * File        : beginner.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-04-05
# * Version     : 0.1.040521
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import tensorflow as tf
from tensorflow import keras


# data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),  # shape = (None, 784)
    tf.keras.layers.Dense(128, activation = "relu"),  # shape = (None, 128)
    tf.keras.layers.Dropout(0.2),  # (None, 128)
    tf.keras.layers.Dense(10, activation = "softmax"),  # (None, 10)
])
print(model.summary())
"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 784)               0         
                                                                 
 dense (Dense)               (None, 128)               100480    
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
Total params: 101,770
Trainable params: 101,770
Non-trainable params: 0
_________________________________________________________________
"""
model.compile(
    optimizer = "adam", 
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"],
)
model.fit(x_train, y_train, epochs = 5)
model.evaluate(x_test, y_test, verbose = 2)




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

