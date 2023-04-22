# -*- coding: utf-8 -*-

# ***************************************************
# * File        : device_config.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-22
# * Version     : 0.1.042223
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import torch

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def try_gpu(i = 0):
    """
    Return gpu(i) if exists, otherwise return cpu().
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')


def try_all_gpus():
    """
    Return all available GPUs, or [cpu(),] if no GPU exists.
    """
    devices = [
        torch.device(f'cuda:{i}') 
        for i in range(torch.cuda.device_count())
    ]

    return devices if devices else [torch.device('cpu')]




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
