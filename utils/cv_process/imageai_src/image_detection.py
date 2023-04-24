# -*- coding: utf-8 -*-


# ***************************************************
# * File        : image_detection.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-14
# * Version     : 0.1.031423
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


"""
https://github.com/fizyr/keras-retinanet/releases/
https://github.com/OlafenwaMoses/ImageAI
https://pjreddie.com/darknet/yolo/
"""

# python libraries
import os
import time
from imageai.Detection import ObjectDetection




start=time.time() #开始计时
execution_path = os.getcmd()
print(f"当前目录是 {os.getcwd()}")

detector = ObjectDetection()
#detector.setModelTypeAsRetinaNet()  #设置算法模型类型为 RetinaNet
#detector.setModelPath(os.path.join(execution_path ,"resnet50_coco_best_v2.1.0.h5"))
#detector.setModelTypeAsTinyYOLOv3() #设置算法模型类型为TinyYOLOv3
#detector.setModelPath( os.path.join(execution_path , "yolo-tiny.h5")) #设置模型文件路径
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path ,"yolo.h5"))
detector.loadModel()
detections=detector.detectObjectsFromImage(input_image=os.path.join(execution_path ,"Lena.png"),output_image_path=os.path.join(execution_path ,"result.png"))
end=time.time() #结束计时
for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"],":",eachObject["box_points"])
print("-----------------------")



# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

