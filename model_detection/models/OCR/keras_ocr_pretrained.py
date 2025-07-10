# -*- coding: utf-8 -*-

# ***************************************************
# * File        : keras_ocr_pretrained.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-29
# * Version     : 0.1.042917
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
import matplotlib.pyplot as plt
import keras_ocr

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# keras-ocr
pipeline = keras_ocr.pipeline.Pipeline()

# images
images = [
    keras_ocr.tools.read(url) for url in [
        'https://upload.wikimedia.org/wikipedia/commons/b/bd/Army_Reserves_Recruitment_Banner_MOD_45156284.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/b/b4/EUBanana-500x112.jpg',
    ]
]
# print(images)

# # detect
prediction_groups = pipeline.recognize(images)

# plot prediction
fig, axs = plt.subplots(nrows = len(images), figsize = (20, 20))
for ax, image, predictions in zip(axs, images, prediction_groups):
    keras_ocr.tools.drawAnnotations(
        image = image, 
        predictions = predictions, 
        ax = ax
    )
plt.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
