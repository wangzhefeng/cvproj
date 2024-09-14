# -*- coding: utf-8 -*-

# ***************************************************
# * File        : torch_hub.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091417
# * Description : description
# * Link        : * https://pytorch.org/hub/
# *             : * https://pytorch.org/docs/stable/hub.html
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

"""
* torch.hub.list()
* torch.hub.help()
* torch.hub.load()
"""


__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.hub as hub

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
