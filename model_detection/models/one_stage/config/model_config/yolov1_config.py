# -*- coding: utf-8 -*-

# ***************************************************
# * File        : yolov1_config.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-10-31
# * Version     : 1.0.103100
# * Description : YOLOv1 Config
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


yolov1_cfg = {
    # input
    'trans_type': 'ssd',
    'multi_scale': [0.5, 1.5],
    # backbone
    'backbone': 'resnet18',
    'pretrained': True,
    'stride': 32,  # P5
    'max_stride': 32,
    # neck
    'neck': 'sppf',
    'expand_ratio': 0.5,
    'pooling_size': 5,
    'neck_act': 'lrelu',
    'neck_norm': 'BN',
    'neck_depthwise': False,
    # head
    'head': 'decoupled_head',
    'head_act': 'lrelu',
    'head_norm': 'BN',
    'num_cls_head': 2,
    'num_reg_head': 2,
    'head_depthwise': False,
    # loss weight
    'loss_obj_weight': 1.0,
    'loss_cls_weight': 1.0,
    'loss_box_weight': 5.0,
    # training configuration
    'trainer_type': 'yolov8',
}




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
