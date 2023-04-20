# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_configure.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-26
# * Version     : 0.1.022616
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
from config.config_loader import settings


def data_configuration(train_ds, validation_ds):
    """
    use buffered prefetching yield data from disk 
    without having I/O becoming blocking

    Args:
        train_ds (_type_): _description_
        validation_ds (_type_): _description_

    Returns:
        _type_: _description_
    """
    train_ds = train_ds.prefetch(buffer_size = settings["DATA"]["buffer_size"])
    validation_ds = validation_ds.prefetch(buffer_size = settings["DATA"]["buffer_size"])
    
    return train_ds, validation_ds



# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

