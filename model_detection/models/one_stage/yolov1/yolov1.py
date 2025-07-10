# -*- coding: utf-8 -*-

# ***************************************************
# * File        : YOLOv1.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-10-31
# * Version     : 1.0.103100
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn

from yolov1_backbone import build_backbone
from yolov1_neck import build_neck

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class YOLOv1(nn.Module):
    
    def __init__(self, cfg, device, input_size, num_classes, trainable, conf_thresh, nms_thresh):
        super(YOLOv1, self).__init__()
        self.cfg = cfg                   # 模型配置文件
        self.device = device             # 设备：CUDA 或 CPU
        # self.input_size = input_size   # 
        self.num_classes = num_classes   # 类别的数量
        self.trainable = trainable       # 训练的标记
        self.conf_thresh = conf_thresh   # 得分阈值
        self.nms_thresh = nms_thresh     # NMS 阈值
        self.stride = 32                 # 网络的最大步长
        # ------------------------------
        # 主干网络(backbone network)
        # ------------------------------
        self.backbone, feat_dim = build_backbone(cfg["backbone"], trainable & cfg["pretrained"])

        # ------------------------------
        # 颈部网络(neck network)
        # ------------------------------
        self.neck = build_neck(cfg, feat_dim, out_dim = 512)
        
        # ------------------------------
        # 检测头(detection head)
        # ------------------------------
        self.head = None
        
        # ------------------------------
        # 预测层(prediction layer)
        # ------------------------------
        self.pred = None
        
    def create_grid(self, input_size):
        """
        用于生成网络坐标矩阵
        """ 
        pass

    def decode_boxes(self, pred):
        """
        解算边界框坐标
        """
        pass
    
    def nms(self, bboxes, scores):
        """
        非极大值抑制操作

        Args:
            bboxes (_type_): _description_
            scores (_type_): _description_
        """
        pass

    def postprocess(self, bboxes, scores):
        """
        后处理，包括得分阈值筛选和 NMS 操作

        Args:
            bboxes (_type_): _description_
            scores (_type_): _description_
        """
        pass

    @torch.no_grad()
    def inference(self, x):
        """
        YOLOv1 前向推理

        Args:
            x (_type_): _description_
        """
        pass

    def forward(self, x, targets = None):
        """
        YOLOv1 的主体运算函数

        Args:
            x (_type_): _description_
            targets (_type_, optional): _description_. Defaults to None.
        """
        pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
