# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_augmentation.py
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
import tensorflow as tf


data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
    ]
)





# 测试代码 main 函数
def main():
    import matplotlib.pyplot as plt
    
    from data_generator import data_generator

    train_ds, validation_ds = data_generator()
    plt.figure(figsize = (10, 10))
    for images, __ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
            plt.show()



if __name__ == "__main__":
    main()

