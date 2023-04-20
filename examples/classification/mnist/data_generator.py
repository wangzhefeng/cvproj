# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_generator.py
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
from config.config_loader import settings


# TODO 数据下载整理
def data_generator():
    """
    生成 Dataset
    """
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        settings["PATH"]["data_root_path"],
        validation_split = settings["DATA"]["validation_split"],
        subset = "training",
        seed = 1337,
        image_size = (settings["IMAGE"]["image_size"]["width"], settings["IMAGE"]["image_size"]["height"]),
        batch_size = settings["MODEL"]["batch_size"],
    )
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
        settings["PATH"]["data_root_path"],
        validation_split = settings["DATA"]["validation_split"],
        subset = "validation",
        seed = 1337,
        image_size = (settings["IMAGE"]["image_size"]["width"], settings["IMAGE"]["image_size"]["height"]),
        batch_size = settings["MODEL"]["batch_size"],
    )

    return train_ds, validation_ds





# 测试代码 main 函数
def main():
    data_generator()


if __name__ == "__main__":
    main()

