# -*- coding: utf-8 -*-


# ***************************************************
# * File        : transfer_learning.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-29
# * Version     : 0.1.032913
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import shutil
import glob

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torchvision import datasets, transforms, models


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# data
# ------------------------------
# data download
img_url, img_path, img_file_name = (
    "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
    "./data/",
    "images.tar",
)
if not os.path.exists(os.path.join(img_path, img_file_name)):
    os.system(f"wget {img_url} ; mv {img_file_name} {img_path} ; tar {img_path}/{img_file_name}")
    os.system(f"rm {img_path}/{img_file_name}")






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
