# -*- coding: utf-8 -*-


# ***************************************************
# * File        : model_building.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-19
# * Version     : 0.1.031923
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


def model_building_API_dnn():
    # model 1
    inputs = tf.keras.layers.Input(shape = (settings["IMAGE"]["flat_image_size"],))
    x = tf.keras.layers.Dense(32, activation = "relu")(inputs)
    output_tensor = tf.keras.layers.Dense(10, activation = "softmax")(x)
    model = tf.keras.models.Model(inputs = inputs, outputs = output_tensor)

    # model 2
    # inputs = tf.keras.layers.Input(shape = (settings["IMAGE"]["flat_image_size"],))
    # x = tf.keras.layers.Dense(64, activation = "relu")(inputs)
    # x = tf.keras.layers.Dense(74, activation = "relu")(x)
    # outputs = tf.keras.layers.Dense(10)(x)
    # model = tf.keras.models.Model(inputs = inputs, outputs = outputs, name = "mnist_dnn_model")

    model.summary()

    return model


def model_building_Seq_dnn():
    # method 1
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape = (28, 28)),
        tf.keras.layers.Dense(128, activation = "relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation = "softmax"),
    ])
    # method 2
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation = "relu", input_shape = (28 * 28, )), # (512,)
        tf.keras.layers.Dense(10, activation = "softmax"),                          # (10,)
    ])

    model.summary()
    tf.keras.utils.plot_model(model, settings["PATH"]["model_image_path"])

    return model


def model_building_Seq_cnn(data_loader):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape = data_loader.input_shape),
        tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(settings["DATA"]["num_classes"], activation='softmax'),
    ])
    
    model.summary()
    tf.keras.utils.plot_model(model, settings["PATH"]["model_image_path"])
    
    return model


def model_building_Seq_cnn(data_loader):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (28, 28, 1)), # (26, 26, 32)
        tf.keras.layers.MaxPooling2D((2, 2)),                                               # (13, 13, 32)
        tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"),                            # (11, 11, 64)
        tf.keras.layers.MaxPooling2D((2, 2)),                                               # (5, 5, 64)
        tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"),                            # (3, 3, 64)
        tf.keras.layers.Flatten(),                                                          # (576,)
        tf.keras.layers.Dense(64, activation = "relu"),                                     # (64,)
        tf.keras.layers.Dense(10, activation = "softmax"),                                  # (10,)
    ])

    model.summary()
    tf.keras.utils.plot_model(model, settings["PATH"]["model_image_path"])

    return model




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

