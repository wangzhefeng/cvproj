# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Averager.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-26
# * Version     : 0.1.042620
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Averager:
    
    def __init__(self) -> None:
        self.current_total = 0.0
        self.iterations = 0.0
    
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
        
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
