# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
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
from config.config_loader import settings
from data_loader import data_loader
from data_generator import data_generator
from model_building import model
from model_training import model_training
from model_predict import model_predict





# 测试代码 main 函数
def main():
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

