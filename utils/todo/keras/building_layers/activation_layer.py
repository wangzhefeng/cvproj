# -*- coding: utf-8 -*-


# ***************************************************
# * File        : activation_layer.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-13
# * Version     : 0.1.031316
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras


"""
from tensorflow.keras import layers
from tensorflow.keras import activations

# ---------------------------
# through activation argument
# ---------------------------
model.add(layers.Dense(64, activation = activations.relu))

model.add(layers.Dense(64, activation = "relu))

# ---------------------------
# through Activation layer
# ---------------------------
model.add(layers.Dense(64))
model.add(layers.Activation(activation.relu))

model.add(layers.Activation("relu"))

# ---------------------------
# custom activations
# ---------------------------
model.add(layers.Dense(64, activation = tf.nn.tanh))
"""




# Layer avtivations
foo = tf.constant(
    [-10, -5, 0.0, 5, 10],
    dtype = tf.float32
)
inputs = tf.random.normal(shape = (32, 10))


relu_res = tf.keras.activations.relu(foo, alpha = 0.0, max_value = None, threshold = 0.0).numpy()
sigmoid_res = tf.keras.activations.sigmoid(foo).numpy()
softmax_res = tf.keras.activations.softmax(inputs, axis = -1)
softplus_res = tf.keras.activations.softplus(foo).numpy()
softsign_res = tf.keras.activations.softsign(foo).numpy()
tanh_res = tf.keras.activations.tanh(foo).numpy()
selu_res = tf.keras.activations.selu(foo).numpy()
elu_res = tf.keras.activations.elu(foo).numpy()
exponential_res = tf.keras.activations.exponential(foo).numpy()
print(relu_res)
print(sigmoid_res)
print(tf.reduce_sum(softmax_res[0, :]))
print(softplus_res)
print(softsign_res)


# Activation layers
relu_layer = tf.keras.layers.ReLU(
    max_value = None, 
    negative_slope = 0.0, 
    threshold = 0.0, 
    # **kwargs,
)
relu_output = relu_layer([-3.0, 1.0, 0.0, 2.0])
print(list(relu_output.numpy()))


softmax_layer = tf.keras.layers.Softmax(
    axis = -1, 
    # **kwargs,
)
softmax_output = softmax_layer(np.asarray([1., 2., 1.])).numpy()
print(softmax_output)
softmax_output = softmax_layer(
    np.asarray([1., 2., 1.]),
    np.asarray([True, False, True], dtype = bool),
).numpy()
print(softmax_output)

leakyrelu_layer = tf.keras.layers.LeakyReLU(
    alpha = 0.3,
    # **kwargs,
)
leakyrelu_output = leakyrelu_layer([-3.0, -1.0, 0.0, 2.0]).numpy()
print(list(leakyrelu_layer))


prelu_layer = tf.keras.layers.PReLU(
    alpha_initializer = "zeros",
    alpha_regularizer = None,
    alpha_constraint = None,
    shared_axes = None,
    # **kwargs,
)



elu_layer = tf.keras.layers.ELU(
    alpha = 1.0, 
    # **kwargs,
)


thresholdedreslu_layer = tf.keras.layers.ThresholdedReLU(
    theta = 1.0, 
    # **kwargs,
)





# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

