# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-04-18
# * Version     : 0.1.041823
# * Description : ORC(光学字符识别)检测图像中的文本，
# *               并在修复过程中填充照片中丢失的部分以生成完整的图像，
# *               删除检测到的文本
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# 1.识别图像中的文本，并使用 Keras OCR 获取每个文本的边界框坐标
# 2.对于每个边界框，应用一个遮罩来告诉算法应该修复图像的哪个部分
# 3.最后，应用一种修复算法对图像的遮罩区域进行修复，从而得到一个无文本图像


# python libraries
import os
import sys


def one_example():
    import matplotlib.pyplot as plt
    import keras_ocr
    
    pipeline = keras_ocr.pipeline.Pipeline()

    # 读取图像
    image_path = None
    img = keras_ocr.tools.read(image_path)
    
    # [(word, box), (word, box), ...]
    prediction_groups = pipeline.recognize([img])

    # 打印图形
    keras_ocr.tools.drawAnnotations(image = img, predictions = prediction_groups[0])
    











# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

