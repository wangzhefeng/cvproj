# -*- coding: utf-8 -*-


# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-08-22
# * Version     : 0.1.082222
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
print(cv2.__version__)


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
GLOBAL_VARIABLE = None


# 创建 VideoCapture 对象
cap = cv2.VideoCapture(0)

flag_collecting = False
images_collected = 0
images_required = 50

# 创建项目目录
directory = os.path.join(os.path.dirname(__file__), "demo")
if not os.path.exists(directory):
    os.mkdir(directory)

# 收集和格式化图像数据集
while True:
    ret, frame = cap.read()

    # 沿 y 轴翻转帧，确保视频以正确的方式显示
    frame = cv2.flip(frame, 1)

    # 设置收集图像的数据量    
    if images_collected == images_required:
        break

    # 绘制黑色矩形
    cv2.rectangle(frame, (380, 80), (620, 320), (0, 0, 0), 3)

    if flag_collecting == True:
        # 提取黑色方块内的切片帧或屏幕的一部分
        sliced_frame = frame[80:320, 380:620]
        # 保存提取的帧
        save_path = os.path.join(directory, f'{images_collected + 1}.jpg')
        cv2.imwrite(save_path, sliced_frame)
        images_collected += 1
    # 显示在给定坐标处收集的图像数量
    cv2.putText(
        frame, 
        f"Saved Images: {images_collected}", 
        (400, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7,
        (0, 0, 0),
        2
    )
    cv2.imshow("Data Collection", frame)
    
    # 键盘字符“s”（用于开始/停止）用于暂停或恢复图像收集，它本质上是一个切换按钮
    k = cv2.waitKey(10)
    if k == ord("s"):
        flag_collecting = not flag_collecting
    # 键盘字符 'q'（用于退出）用于关闭窗口
    if k == ord("q"):
        break

print(images_collected, "images saved to directory")
cap.release()
cv2.destroyAllWindows()




# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

