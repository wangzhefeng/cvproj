# -*- coding: utf-8 -*-

# ***************************************************
# * File        : run_lenet5.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-02-24
# * Version     : 1.0.022423
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = str(os.getcwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

from exps.exp_classification import Exp_Classification
from utils.cv.args_tools import DotDict

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]




# 测试代码 main 函数
def main():
    # params
    args = {
        "model_name": "lenet5",
        "data_name": "mnist",
        "use_valid": 0,
        "use_gpu": 1,
        "gpu_type": "mps",
        "use_multi_gpu": 0,
        "devices": "0,1,2,3",
        "batch_size": 64,
        "in_channels": 1,
        "num_classes": 10,
        "num_epochs": 10,
        "algo": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.1,
    }
    args = DotDict(args)
    # exp
    exp = Exp_Classification(args)
    # training
    exp.train()
    # test
    exp.test()

if __name__ == "__main__":
    main()
