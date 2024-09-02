# -*- coding: utf-8 -*-

# ***************************************************
# * File        : example_image.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-25
# * Version     : 0.1.042519
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
from pathlib import Path
from PIL import Image

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def get_example_image(img_name = 'park.jpg'):
    """
    name can be bus.jpg / park.jpg / zidane.jpg
    """
    path = Path(__file__)
    print(path.parent.parent)
    img_path = str(path.parent.parent/f"images/{img_name}")
    assert os.path.exists(img_path), 'img_name can only be bus.jpg / park.jpg / zidane.jpg'

    return Image.open(img_path)




# 测试代码 main 函数
def main():
    img = get_example_image(img_name = "lena.jpg")
    print(img)

if __name__ == "__main__":
    main()
