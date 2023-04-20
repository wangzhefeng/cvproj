# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_preprocessing.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-24
# * Version     : 0.1.022423
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
from typing import Tuple
import tensorflow as tf
from data_augmentation import data_augmentation


def data_preprocessing_GPU(input_shape: Tuple(int, int)):
    """
    data augmentation will happen on device, 
    synchronously with the rest of the model execution, 
    meaning that it will benefit from GPU acceleration.

    Args:
        input_shape (Tuple): _description_

    Returns:
        _type_: _description_
    """
    inputs = tf.keras.Input(shape = input_shape)
    x = data_augmentation(inputs)
    x = tf.keras.layers.Rescaling(1. / 255)(x)
    
    return x


def data_preprocessing_CPU(train_ds):
    """
    data augmentation will happen on CPU, asynchronously, 
    and will be buffered before going into the model.
    If you're training on CPU, this is the better option, 
    since it makes data augmentation asynchronous and non-blocking.

    Args:
        train_ds (_type_): _description_

    Returns:
        _type_: _description_
    """
    augmented_train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training = True), y)
    )

    return augmented_train_ds






# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

