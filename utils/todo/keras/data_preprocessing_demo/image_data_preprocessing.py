# -*- coding: utf-8 -*-


# ***************************************************
# * File        : image_data_preprocessing.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-23
# * Version     : 0.1.022323
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # or any {"0", "1", "2"}

# import logging
# logging.getLogger("tensorflow").disabled = True

# import tensorflow as tf
# tf.get_logger().setLevel("INFO")

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ====================================================
# data preprocessing
# ====================================================
# Tokenization of string data, followed token indexing
# Feature normalization
# Rescaling the data to small values
# ------------------------------------------
# image
# ------------------------------------------
training_data = np.random.randint(0, 256, size = (64, 200, 200, 3)).astype("float32")

# normalizing features
normalizer = keras.layers.Normalization(axis = -1)
normalizer.adapt(training_data)
normalizer_data = normalizer(training_data)
print(normalizer_data)
print("var: %.4f" % np.var(normalizer_data))
print("mean: %.4f" % np.mean(normalizer_data))

# rescaling and center-cropping images
cropper = keras.layers.CenterCrop(height = 150, width = 150)
scaler = keras.layers.Rescaling(scale = 1.0 / 255)
output_data = scaler(cropper(training_data))
print(output_data)
print("shape:", output_data.shape)
print("min", np.min(output_data))
print("max", np.max(output_data))

