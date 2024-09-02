# -*- coding: utf-8 -*-

# ***************************************************
# * File        : open_cv_pixel_gray.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-03-31
# * Version     : 0.1.033122
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

import numpy as np
import cv2

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


img = cv2.imread("D:/projects/computer_vision/cvproj/data/images/lena_gray.bmp", 0)
cv2.imshow("before", img)

print(f"img 尺寸：{img.shape}")
print(f"img[50, 90] 原始值：{img[50, 90]}")
img[10:100, 80:100] = 255
print(f"img[50, 90] 修改值：{img[50, 90]}")
cv2.imshow("after", img)


cv2.waitKey()
cv2.destroyAllWindows()





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
