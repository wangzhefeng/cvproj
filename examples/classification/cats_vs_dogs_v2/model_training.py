# -*- coding: utf-8 -*-


# ***************************************************
# * File        : model_training.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-26
# * Version     : 0.1.022617
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import tensorflow as tf
from config.config_loader import settings


def model_training(model, train_ds, validation_ds):
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(settings["PATH"]["model_path"], "save_at_{epoch}.h5")
        ),
    ]
    # 模型编译
    model.compile(
        optimizer = tf.keras.optimizers.Adam(1e-3),
        loss = "binary_crossentropy",
        metrics = ["accuracy"],
    )
    # 模型训练
    model.fit(
        train_ds,
        epochs = settings["MODEL"]["epochs"],
        callbacks = callbacks,
        validation_data = validation_ds,
    )
    # 模型保存
    model.save()
    return model




# 测试代码 main 函数
def main():
    from data_loader import data_loader
    from data_generator import data_generator
    from model_building import model
    
    # 1.去除异常格式的图片数据
    data_loader()
    # 2.生成训练和验证数据集
    train_ds, validation_ds = data_generator()
    # 3.模型训练
    trained_model = model_training(model, train_ds, validation_ds)


if __name__ == "__main__":
    main()

