import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


"""
- 核心 preprocessing layers
    - TextVectorization
    - Normalization
- 结构化数据 preprocessing layers
    - CategoryEncoding
    - Hashing
    - Discretization
    - StringLookup
    - IntegerLookup
    - CategoryCrossing
- 图像数据 preprocessing layers
    - Resizing
    - Rescaling
    - CenterCrop
- 图像数据增强 augmentation layers
    - RandomCrop
    - RandomFlip
    - RandomTranslation
    - RandomRotation
    - RandomZoom
    - RandomHeight
    - RandomWidth



- adapt() method
- preprocessing data
    - before model
    - inside model
- 技巧
    - image
        - image data augmentation
    - numeric
        - normalizing numeric feature
    - categorical
        - encoding string categorical features via one-hot encoding
        - encoding integer categorical features via one-hot encoding
        - hashing on integer categorical feature
    - text
        - encoding text as a sequence of token indices
        - encoding text as a dense matrix of ngrams with multi-hot encoding
        - encoding text as a dense matrix of ngrams with TF-IDF weighting
"""

# ========================================
# adapt() method
# ========================================





# ========================================
# 图像数据增强
# ========================================
data_augmentation = keras.Sequential([
    preprocessing.RandomFlip("horizontal"),
    preprocessing.RandomRotation(0.1),
    preprocessing.RandomZoom(0.1),
])

input_shape = (32, 32, 3)
classes = 10

inputs = keras.Input(shape = input_shape)
x = data_augmentation(inputs)
x = preprocessing.Rescaling(1.0 / 255)(x)
outputs = keras.applications.ResNet50(weights = None, input_shape = input_shape, classes = classes)(x)

model = keras.Model(inputs, outputs)
