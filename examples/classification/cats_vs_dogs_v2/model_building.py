# -*- coding: utf-8 -*-


# ***************************************************
# * File        : model.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-26
# * Version     : 0.1.022616
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import tensorflow as tf
from tensorflow import keras
from data_augmentation import data_augmentation
from config.config_loader import settings


def make_Xception_model(input_shape, num_classes):
    # inputs
    inputs = tf.keras.Input(shape = input_shape)

    # image augmentation block
    x = data_augmentation(inputs)

    # entry block
    x = tf.keras.layers.Rescaling(1.0 / 255)(x)

    x = tf.keras.layers.Conv2D(32, 3, strides = 2, padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64, 3, padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # set aside residual
    previous_block_activation = x

    for size in [128, 256, 512, 728]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding = "same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding = "same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides = 2, padding = "same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(size, 1, strides = 2, padding = "same")(previous_block_activation)
        x = tf.keras.layers.add([x, residual])  # add back residual
        
        previous_block_activation = x  # set aside nex residual

    x = tf.keras.layers.SeparableConv2D(1024, 3, padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes
    
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(units, activation = activation)(x)
    
    return keras.Model(inputs, outputs)


model = make_Xception_model(
    input_shape = (
        settings["IMAGE"]["image_size"]["width"], 
        settings["IMAGE"]["image_size"]["height"]
    ) + (3,),
    num_classes = 2,
)


# 测试代码 main 函数
def main():
    model.summary()
    tf.keras.utils.plot_model(model, to_file = settings["PATH"]["model_path"], show_shapes = True)


if __name__ == "__main__":
    main()

