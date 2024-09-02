# -*- coding: utf-8 -*-

# ***************************************************
# * File        : opencv_channel.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-04-01
# * Version     : 0.1.040123
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


lena = cv2.imread("D:/projects/computer_vision/cvproj/data/images/lena.png")
# cv2.imshow("lean", lena)
# b = lena[:, :, 0]
# g = lena[:, :, 1]
# r = lena[:, :, 2]

# cv2.imshow("b", b)
# cv2.imshow("g", g)
# cv2.imshow("r", r)


# lena[:, :, 0] = 0
# cv2.imshow("lenab0", lena)

# lena[:, :, 1] = 0
# cv2.imshow("lenab0g0", lena)

b, g, r = cv2.split(lena)

bgr = cv2.merge([b, g, r])
rgb = cv2.merge([r, g, b])

cv2.imshow("lena", lena)
cv2.imshow("bgr", bgr)
cv2.imshow("rgb", rgb)


cv2.waitKey()
cv2.destroyAllWindows()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
