# -*- coding: utf-8 -*-

# ***************************************************
# * File        : MNIST_tf.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-25
# * Version     : 0.1.042500
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# --------------------------------------------------------
# Load MNIST data to tf.data.Dataset
# --------------------------------------------------------
(train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
train_data = np.expand_dims(
    train_data.astype(np.float32) / 255.0, 
    axis = -1
)
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
# for image, label in mnist_dataset:
#     plt.title(label.numpy())
#     plt.imshow(image.numpy()[:, :, 0])
#     plt.show()


# --------------------------------------------------------
# tf.data.Dataset 数据预处理
# --------------------------------------------------------
# Dataset.map(f)
# Dataset.shuffle(buffer_size)
# Dataset.batch(batch_size)
# Dataset.repeat()
# Dataset.reduce()
# Dataset.take()
# Dataset.prefetch()


def rot90(image, label):
    image = tf.image.rot90(image)
    return image, label

mnist_dataset = mnist_dataset.map(rot90)

for image, label in mnist_dataset:
    plt.title(label.numpy())
    plt.imshow(image.numpy()[:, :, 0])
    plt.show()





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
