# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-08-23
# * Version     : 0.1.082322
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None

# 黑色图像模板
img_blank = np.zeros(shape = (512, 512, 3), dtype = np.int16)
# plt.imshow(img_blank)
# plt.show()

# 直线
# line_red = cv2.line(
#     img = img_blank,
#     pt1 = (0, 0),
#     pt2 = (511, 511),
#     color = (255, 0, 0),
#     thickness = 5,
#     lineType = 8,
# )
# plt.imshow(line_red)
# plt.show()


# 矩形
# img_rectangle = cv2.rectangle(
#     img = img_blank, 
#     pt1 = (384, 0),
#     pt2 = (510, 128), 
#     color = (0, 0, 255), 
#     thickness = 5,
#     lineType = 8,
# )
# plt.imshow(img_rectangle)
# plt.show()

# 圆圈
# img_circle = cv2.circle(
#     img = img_blank, 
#     center = (447, 63), 
#     radius = 63, 
#     color = (0, 0, 255), 
#     thickness = -1,
#     lineType = 8,
# )
# plt.imshow(img_circle)
# plt.show()

# 文字
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
img_text = cv2.putText(
    img = img_blank,             # 图像
    text = "OpenCV",             # 文字内容
    org = (150, 200),              # 文字坐标
    fontFace = font,             # 文字字体
    fontScale = 2,               # 文字比例
    color = (255, 255, 255),     # 文字颜色
    thickness = 5,               # 文字字体粗细
    lineType = cv2.LINE_AA,      # 文字线条类型
)
plt.imshow(img_text)
plt.show()




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

