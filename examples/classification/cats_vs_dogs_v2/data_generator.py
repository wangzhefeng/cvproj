# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_generator.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-24
# * Version     : 0.1.022422
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # or any {"0", "1", "2"}
import tensorflow as tf
from config.config_loader import settings


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




def main():
    from data_loader import data_loader
    
    # 1.去除异常格式的图片数据
    data_loader()
    # 2.生成训练和验证数据集
    train_ds, validation_ds = data_generator()


if __name__ == "__main__":
    main()

