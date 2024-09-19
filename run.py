# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-17
# * Version     : 1.0.091701
# * Description : description
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

import argparse

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]






# 测试代码 main 函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument()

if __name__ == "__main__":
    main()
