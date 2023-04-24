# -*- coding: utf-8 -*-


# ***************************************************
# * File        : text_data_preprocessing.py
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
# text
# ------------------------------------------
training_data = np.array([
    ["This is the 1st sample."], 
    ["And here's the 2nd sample."]
])

# turning strings into sequences of integer word indices
vectorizer = keras.layers.TextVectorization(output_mode = "int")
vectorizer.adapt(training_data)
integer_data = vectorizer(training_data)
print(integer_data)

# turning string into sequences of one-hot encoded bigrams
vectorizer = keras.layers.TextVectorization(output_mode = "binary", ngrams = 2)
vectorizer.adapt(training_data)
integer_data = vectorizer(training_data)
print(integer_data)

