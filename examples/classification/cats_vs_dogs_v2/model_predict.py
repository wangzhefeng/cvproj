# -*- coding: utf-8 -*-


# ***************************************************
# * File        : model_predict.py
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
import matplotlib.pyplot as plt
import tensorflow as tf
from config.config_loader import settings


# TODO 处理成 API 形式
def model_predict(image_path, show_image, model):
    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size = (
            settings["image"]["image_size"]["width"], 
            settings["image"]["image_size"]["height"]
        ),
    )
    if show_image:
        plt.figure(figsize = (10, 10))
        plt.imshow(img)
        plt.title(label = None)
        plt.axis("off")
        plt.show()
    
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = predictions[0]
    print(
        "this image is %.2f percent cat and %.2f percent dog."
        % (100 * (1 - score), 100 * score)
    )
    return score
    



# 测试代码 main 函数
def main():
    from data_loader import data_loader
    from data_generator import data_generator
    from model_building import model
    from model_training import model_training
    
    # 1.去除异常格式的图片数据
    data_loader()
    # 2.生成训练和验证数据集
    train_ds, validation_ds = data_generator()
    # 3.模型训练
    trained_model = model_training(model, train_ds, validation_ds)
    # 4.模型载入
    # TODO
    # 5.模型预测
    score = model_predict(
        image_path = os.path.join(settings["path"]["data_root_path"], "cat/6779.jpg"),
        show_image = False,
        model = trained_model,
    )


if __name__ == "__main__":
    main()

