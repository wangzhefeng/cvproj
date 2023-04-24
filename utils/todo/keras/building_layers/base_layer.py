# -*- coding: utf-8 -*-


# ***************************************************
# * File        : base_layer.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-13
# * Version     : 0.1.031310
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import tensorflow as tf
from tensorflow import keras


"""
# API
tf.keras.layers.Layers(
    trainable = True,
    name = None,
    dtype = None,
    dynamic = False,
    **kwargs,
)
"""


class SimpleDense(keras.Layer):

    def __init__(self, units = 32):
        super(SimpleDense, self).__init__()
        self.units = units

    def build(self, input_shape):
        """
        Create the state of the layer(weight)

        :param input_shape: _description_
        :type input_shape: _type_
        """
        # method 1
        # w_init = tf.random_normal_initializer()
        # self.w = tf.Variable(
        #     initial_value = w_init(shape = (input_shape[-1], self.units),
        #                            dtype = "float32"),
        #     trainable = True,
        # )
        # method 2
        self.w = self.add_weight(
            shape = (input_shape[-1], self.units),
            initializer = "random_normal",
            trainable = True
        )
        
        # method 1
        # b_init = tf.zeros_initializer()
        # self.b = tf.Variable(
        #     initial_value = b_init(shape = (self.units,), dtype = "float32"),
        #     trainable = True,
        # )
        # method 2
        self.b = self.add_weight(
            shape = (self.units,),
            initializer = "random_normal",
            trainable = True
        )
    
    def call(self, inputs):
        """
        Defines the computation from inputs to outputs

        :param inputs: _description_
        :type inputs: _type_
        :return: _description_
        :rtype: _type_
        """
        return tf.matmul(inputs, self.w) + self.b
    
    def get_config(self):
        pass




# 测试代码 main 函数
def main():
    # Instantiates the layer
    linear_layer = SimpleDense(4)

    # call `build(input_shape)` and create the weights
    y = linear_layer(tf.ones((2, 2)))
    
    # layer weight
    assert len(linear_layer.weight) == 2

    # These weights are trainable, so they're listed in `trainable_weights`
    assert len(linear_layer.trainable_weights) == 2

    linear_layer.non_trainable_weights
    linear_layer.trainable
    linear_layer.get_weights()
    linear_layer.set_weights()
    linear_layer.get_config()
    linear_layer.add_loss()
    linear_layer.add_metric()
    linear_layer.losses
    linear_layer.metrics
    linear_layer.dynamic

if __name__ == "__main__":
    main()

