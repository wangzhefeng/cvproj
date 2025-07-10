# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_facory.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091419
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def data_provider(flag: str):
    # params
    shuffle = True if flag == "train" else False
    drop_last = True if flag == "train" else False




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
