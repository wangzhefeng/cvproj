# -*- coding: utf-8 -*-

# ***************************************************
# * File        : opencv_pixel.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-03-31
# * Version     : 0.1.033121
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

import numpy as np
import cv2

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


# 生成 8×8 的黑色图像
img = np.zeros((256, 256), dtype = np.uint8)
print(f"img=\n{img}")
cv2.imshow("one", img)

print(f"读取像素点 img[0, 3]={img[0, 3]}")
img[0:30, 60:90] = 255
print(f"修改后 img=\n{img}")

print(f"读取修改后像素点 img[0, 3]={img[0, 3]}")
cv2.imshow("two", img)

cv2.waitKey()
cv2.destroyAllWindows()


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
