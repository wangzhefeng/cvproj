# -*- coding: utf-8 -*-

# ***************************************************
# * File        : opencv_test.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-03-30
# * Version     : 0.1.033016
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import cv2

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


lena = cv2.imread("D:/projects/computer_vision/cvproj/data/images/lena.jpg", )
print(type(lena))
print(lena.shape)
print(lena)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
