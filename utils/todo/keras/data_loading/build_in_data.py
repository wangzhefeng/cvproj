# -*- coding: utf-8 -*-


# ***************************************************
# * File        : build_in_datasets_loading.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-12
# * Version     : 0.1.031223
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import tensorflow as tf
from tensorflow import keras


def mnist_loading():
    """
    数据保存目录: ~/.keras/datasets/mnist.npz

    :return: _description_
    :rtype: _type_
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path = "mnist.npz")
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    return (x_train, y_train), (x_test, y_test)


def cifar10_loading():
    """
    数据保存目录: 

    :return: _description_
    :rtype: _type_
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    return (x_train, y_train), (x_test, y_test)


def cifar100_loading(label_mode = "fine"):
    """
    数据保存目录:

    :param label_mode: 数据标签， fine: 细粒度标, coarse: 粗粒度, defaults to "fine"
    :type label_mode: str, optional
    :return: _description_
    :rtype: _type_
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode = label_mode)
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)
    
    return (x_train, y_train), (x_test, y_test)








# 测试代码 main 函数
def main():
    # cifar10_loading()
    cifar100_loading()


if __name__ == "__main__":
    main()

