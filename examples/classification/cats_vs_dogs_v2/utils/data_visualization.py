# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_visualize.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-24
# * Version     : 0.1.022423
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
_path = os.path.abspath(os.path.dirname(__file__))
if os.path.join(_path, "..") in sys.path:
    pass
else:
    sys.path.append(os.path.join(_path, ".."))

import matplotlib.pyplot as plt


def data_visualization(train_ds):
    plt.figure(figsize = (10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
            plt.show()




# 测试代码 main 函数
def main():
    from data_loader import data_loader
    from data_generator import data_generator
    
    # 1.数据载入
    data_loader()
    # 2.生成训练和验证数据集
    train_ds, validation_ds = data_generator()
    # 3.数据查看
    data_visualization(train_ds)


if __name__ == "__main__":
    main()

