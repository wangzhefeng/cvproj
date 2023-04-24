# -*- coding: utf-8 -*-


# ***************************************************
# * File        : text_data_loading.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-23
# * Version     : 0.1.022322
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import numpy as np
import tensorflow as tf
from tensorflow import keras


# ====================================================
# data loading
# ====================================================
# ----------------------------------------------------
# Numpy arrays
# Tensorflow Dataset objects
# Python generators

# tf.keras.preprocessing.image_dataset_from_directory
# tf.keras.preprocessing.text_dataset_from_directory
# tf.data.experimental.make_csv_dataset
# ----------------------------------------------------
# create a dataset
dataset = keras.preprocessing.text_dataset_from_directory(
    "path/to/main_directory",
    batch_size = 64,
)
# iterate over the batches yielded by the dataset
for data, labels in dataset:
    print(data.shape)  # (64,)
    print(data.dtype)  # string
    print(labels.shape)  # (64,)
    print(labels.dtype)  # int32
