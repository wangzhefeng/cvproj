# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-29
# * Version     : 0.1.032920
# * Description : description
# * Link        : https://github.com/Aman-Preet-Singh-Gulati/face-detection-CV
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageUtils:
    """
    图像处理工具类
    """

    def __init__(self) -> None:
        pass

    def image_load(self, image_path: str, is_show: bool = False):
        """
        图像加载

        Args:
            image_path (str): 图像地址
            is_show (bool, optional): 是否展示图像. Defaults to False.
        """
        # 读取图像
        image = cv2.imread(image_path)
        # 图像显示
        if is_show:
            self.image_show(image, cmap = None)

        return image

    @staticmethod
    def image_show(image, cmap = None) -> None:
        """
        图像展示

        Args:
            cmap (_type_): _description_
        """
        if image is not None:
            plt.imshow(image, cmap = cmap)
            plt.show()

    def BGR_2_GRAY(self, image, is_show: bool = False):
        """
        将 BGR 格式彩色图像转换为灰色图像
        """
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if is_show:
            self.image_show(image, cmap = "gray")

        return image

    def BGR_2_RGB(self, image, is_show: bool = False):
        """
        将 BGR 格式彩色图像转换为 RGB 格式的彩色图像:
          OpenCV 检测器功能默认读取 BGR 格式的现有图像, 
          但最终用户通常不会考虑 BGR 格式, 因此需要将 BGR
          格式的图像转换为 RGB 格式，即彩色图像(color)
        """
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if is_show:
            self.image_show(image, cmap = None)

        return image

    @staticmethod
    def load_face_classifier(haar_cascade_face_path: str):
        """
        HAAR 级联是计算机视觉领域的一个很好的术语。
        当我们谈论 HAAR 级联分类器时，不仅仅是人脸预训练分类器，
        我们可以得到经过训练来检测微笑、汽车、公共汽车的分类器，
        这些级联文件总是采用 XML 文件格式，通常我们使用现有的级联文件，
        但事实上我们也可以在这里创建它们

        Args:
            haar_cascade_face_path (str): _description_

        Returns:
            _type_: _description_
        """
        face_classifier = cv2.CascadeClassifier(haar_cascade_face_path)
        
        return face_classifier

    @staticmethod
    def detect_face_coord(image, 
                          face_classifier, 
                          scaleFactory: float, 
                          minNeighbors: float) -> np.ndarray:
        face_coord = face_classifier.detectMultiScale(
            image, 
            scaleFactor = scaleFactory, 
            minNeighbors = minNeighbors,
        )

        return face_coord

    def plot_face_rect(self, 
                       image, 
                       face_coord: np.ndarray, 
                       is_show: bool = False,
                       cmap: str = None) -> None:
        for (x_face, y_face, w_face, h_face) in face_coord:
            cv2.rectangle(
                image, 
                (x_face, y_face), (x_face + w_face, y_face + h_face),
                (0, 255, 0),
                2
            )
        if is_show:
            image = self.BGR_2_RGB(image)
            self.image_show(image, cmap)

    def detect_face(self, 
                    image, 
                    haar_cascade_face_path, 
                    scaleFactor = 1.1, 
                    minNeighbors = 1,
                    image_save_path: str = None):
        """
        自动人脸检测过程

        Args:
            cascade (_type_): _description_
            image (_type_): _description_
            scaleFactor (float, optional): _description_. Defaults to 1.1.
        """
        # 图片加载
        image_copy = image.copy()
        # 图片转换为灰度图像，方便 cv2 读取
        image_gray = self.BGR_2_GRAY(image_copy)
        # 加载人脸正面分类器
        face_classifier = self.load_face_classifier(haar_cascade_face_path)
        # 人脸坐标检测
        face_coord = self.detect_face_coord(
            image_gray, 
            face_classifier = face_classifier, 
            scaleFactory = scaleFactor,
            minNeighbors = minNeighbors
        )
        print(f"Faces found: {len(face_coord)}")
        # 人脸坐标矩形绘制
        self.plot_face_rect(
            image_copy, 
            face_coord = face_coord,
            is_show = True,
        )
        # 图像保存
        if image_save_path:
            cv2.imwrite(image_save_path, image_copy)

        return image_copy




# 测试代码 main 函数
def main():
    dirname = os.path.dirname(__file__)
    print(dirname)

    # haar 级联文件
    haar_cascade_face_path = os.path.join(
        dirname, 
        "haarcascades/haarcascade_frontalface_alt2.xml"
    )

    # Imageutils 实例
    image_utils = ImageUtils()

    # 测试 baby1.png 图像
    img_path = os.path.join(dirname, "data/origin/baby1.png")
    img_save_path = os.path.join(dirname, "data/detected/baby1.png")
    image = image_utils.image_load(image_path = img_path, is_show = False)
    image_utils.detect_face(
        image = image, 
        haar_cascade_face_path = haar_cascade_face_path, 
        image_save_path = img_save_path
    )

    # # 测试 bagy2.png 图像
    # test_img_path = os.path.join(dirname, "data/origin/baby2.png")
    # test_img_save_path = os.path.join(dirname, "data/detected/baby2.png")
    # test_face_image = image_utils.image_load(image_path = test_img_path, is_show = False)
    # image_utils.detect_face(
    #     image = test_face_image, 
    #     haar_cascade_face_path = haar_cascade_face_path,
    #     image_save_path = test_img_save_path,
    # )

    # # 测试多脸图像 group.png
    # group_img_path = os.path.join(dirname, "data/origin/group.png")
    # group_img_save_path = os.path.join(dirname, "data/detected/group.png")
    # group_face_image = image_utils.image_load(image_path = group_img_path, is_show = False)
    # image_utils.detect_face(
    #     group_face_image, 
    #     haar_cascade_face_path,
    #     image_save_path = group_img_save_path,
    # )

if __name__ == "__main__":
    main()
