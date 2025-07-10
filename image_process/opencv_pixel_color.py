# -*- coding: utf-8 -*-

# ***************************************************
# * File        : opencv_pixel_color.py
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
from pathlib import Path
ROOT = str(Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
import cv2

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


img = cv2.imread("E:/projects/computer_vision/cvproj/dataset/images/lena.png")
cv2.imshow("before", img)

print(f"访问 img[0, 0] = {img[0, 0]}")
print(f"访问 img[0, 0, 0] = {img[0, 0, 0]}")
print(f"访问 img[0, 0, 1] = {img[0, 0, 1]}")
print(f"访问 img[0, 0, 2] = {img[0, 0, 2]}")
print(f"访问 img[50, 0] = {img[50, 0]}")
print(f"访问 img[100, 0] = {img[100, 0]}")

# 区域 1：白色
img[0:50, 0:100, 0:3] = 255

# 区域 2：灰色
img[50:100, 0:100, 0:3] = 128

# 区域 3：黑色
img[100:150, 0:100, 0:3] = 0

# 区域 4：红色
img[150:200, 0:100] = [0, 0, 255]

cv2.imshow("after", img)
print(f"修改后 img[0, 0] = {img[0, 0]}")
print(f"修改后 img[0, 0, 0] = {img[0, 0, 0]}")
print(f"修改后 img[0, 0, 1] = {img[0, 0, 1]}")
print(f"修改后 img[0, 0, 2] = {img[0, 0, 2]}")
print(f"修改后 img[50, 0] = {img[50, 0]}")
print(f"修改后 img[100, 0] = {img[100, 0]}")

cv2.waitKey()
cv2.destroyAllWindows()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
