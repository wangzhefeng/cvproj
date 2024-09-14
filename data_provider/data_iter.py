# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_iter.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091417
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

import numpy as np
import torch

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def data_iter(features: torch.Tensor, labels: torch.Tensor, batch_size: int = 8):
    """
    构建数据管道迭代器
    """
    num_examples = len(features)  # 样本数量
    indices = list(range(num_examples))  # 样本索引列表
    np.random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        indexs = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])

        yield features.index_select(0, indexs), labels.index_select(0, indexs)




# 测试代码 main 函数
def main():
    # 样本数量
    num_sample = 400
    # 生成测试用数据集
    X = 10 * torch.rand([num_sample, 2]) - 5.0  # torch.rand 是均匀分布
    w0 = torch.tensor([[2.0], [-3.0]])
    b0 = torch.tensor([[10.0]])
    Y = X@w0 + b0 + torch.normal(0.0, 2.0, size = [num_sample, 1])

    # 测试数据管道效果
    batch_size = 8
    (features, labels) = next(data_iter(X, Y, batch_size))
    print(features)
    print(labels)

if __name__ == "__main__":
    main()
