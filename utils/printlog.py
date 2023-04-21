# -*- coding: utf-8 -*-


# ***************************************************
# * File        : printlog.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-04
# * Version     : 0.1.040417
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import datetime


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def printlog(info):
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 80 + f"{nowtime}")
    print(str(info), "\n")




# 测试代码 main 函数
def main():
    info = "test string"
    printlog(info)

if __name__ == "__main__":
    main()
