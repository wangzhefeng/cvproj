# -*- coding: utf-8 -*-


# ***************************************************
# * File        : model_evaluate.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-20
# * Version     : 0.1.032000
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys


def model_evaluate(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)
    test_scores = model.evaluate(x_test, y_test, verbose = 2)
    
    return test_loss, test_acc, test_scores


# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

