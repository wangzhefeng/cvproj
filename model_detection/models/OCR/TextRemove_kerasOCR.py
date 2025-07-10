# -*- coding: utf-8 -*-

# ***************************************************
# * File        : keras_ocr.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-29
# * Version     : 0.1.042914
# * Description : ORC(光学字符识别)检测图像中的文本，
# *               并在修复过程中填充照片中丢失的部分以生成完整的图像，
# *               删除检测到的文本
# *               1.识别图像中的文本，并使用 Keras OCR 获取每个文本的边界框坐标
# *               2.对于每个边界框，应用一个遮罩来告诉算法应该修复图像的哪个部分
# *               3.最后，应用一种修复算法对图像的遮罩区域进行修复，从而得到一个无文本图像
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras_ocr

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def line_mask_params(box):
    """
    线遮罩参数，线的起点和终点，线的厚度
    """
    x1, y1 = box[1][0]
    x2, y2 = box[1][1]
    x3, y3 = box[1][2]
    x4, y4 = box[1][3]
    # 线的起点: 左上角和左下角之间的中点
    x_mid_start, y_mid_start = int((x2 + x3) / 2), int((y2 + y3) / 2)
    # 线的终点: 右上角和右下角之间的中点
    x_mid_end, y_mid_end = int((x1 + x4) / 2), int((y1 + y4) / 2)
    start_point = (x_mid_start, y_mid_start)
    end_point = (x_mid_end, y_mid_end)
    
    # 线的厚度: 左上角和左下角之间的线长度
    thickness = int(math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2))
    # thickness = int(math.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2))
    
    return start_point, end_point, thickness


def inpaint_text(img_path, pipeline, remove_list = []):
    """
    _summary_

    Args:
        img_path (_type_): _description_
        remove_list (_type_): _description_
        pipeline (_type_): _description_

    Returns:
        _type_: _description_
    """
    # ------------------------------
    # Keras OCR detect
    # ------------------------------
    # 输入图像，包含要删除的文本
    img = keras_ocr.tools.read(img_path)
    # 生成 [(word, box), (word, box), ...], 包含四个角的坐标 (x, y)
    # 数组的第一个元素对应左上角的坐标，第二个元素对应右下角，第三个元素是右上角，第四个元素是左下角
    prediction_groups = pipeline.recognize([img])
    # 打印图形
    keras_ocr.tools.drawAnnotations(image = img, predictions = prediction_groups[0])
    # ------------------------------
    # OpenCV image restore
    # ------------------------------
    # line mask for word
    for box in prediction_groups[0]:
        if remove_list == [] or box[0] in remove_list:
            # ------------------------------
            # 线遮罩: 更灵活地覆盖不同方向的文本
            #    - 矩形遮罩只适用于平行或垂直于 x 轴的单词
            #    - 圆形遮罩将覆盖比较大的区域
            # ------------------------------
            start_point, end_point, thickness = line_mask_params(box)
            mask = np.zeros(img.shape[:2], dtype = "uint8")
            cv2.line(
                img = mask, 
                pt1 = start_point, 
                pt2 = end_point, 
                color = 255, 
                thickness = thickness,
            )
            # 检查遮罩区域，确保其正常工作
            masked = cv2.bitwise_and(src1 = img, src2 = img, mask = mask)
            plt.imshow(masked)
            # ------------------------------
            # 使用 OpenCV 应用修复算法，需要提供两幅图像：
            # 1.输入图像，包含我们要删除的文本。
            # 2.遮罩图像，它显示图像中要删除的文本在哪里。第二个图像的尺寸应与输入的尺寸相同
            # ------------------------------
            # 图像修复
            img_inpainted = cv2.inpaint(
                src = img, 
                inpaintMask = mask, 
                inpaintRadius = 7, 
                flags = cv2.INPAINT_NS
            )
            # 保存图像
            img_inpainted_rgb = cv2.cvtColor(src = img_inpainted, code = cv2.COLOR_BGR2RGB)
            cv2.imwrite("text_free_image.jpg", img_inpainted_rgb)
    
    return img_inpainted




# 测试代码 main 函数
def main():
    img_path = str(Path(__file__).parent.parent/"example.jpg")
    pipeline = keras_ocr.pipeline.Pipeline()
    inpaint_text(img_path, pipeline)

if __name__ == "__main__":
    main()
