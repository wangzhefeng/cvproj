# -*- coding: utf-8 -*-

# ***************************************************
# * File        : tf_numpy_dataset.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-25
# * Version     : 0.1.042500
# * Description : Transform numpy or tensorflow array to tf.data.Dataset
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import numpy as np
import tensorflow as tf

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# numpy variable
X_numpy = np.array([2013, 2014, 2015, 2016, 2017])
Y_numpy = np.array([12000, 14000, 15000, 16500, 17500])

# tensorflow variable
X_tf = tf.constant([2013, 2014, 2015, 2016, 2017])
Y_tf = tf.constant([12000, 14000, 15000, 16500, 17500])


dataset = tf.data.Dataset.from_tensor_slices((X_numpy, Y_numpy))
for x, y in dataset:
    print(x.numpy(), y.numpy())





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
