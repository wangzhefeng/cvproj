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
import os
import torch
import tensorflow as tf

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


def get_gpus_cpus():
    """
    获得当前主机上某种特定运算设备类型(GPU,CPU)的列表
    """
    gpus = tf.config.list_physical_devices(device_type = "GPU")
    cpus = tf.config.list_physical_devices(device_type = "CPU")
    print(gpus, cpus)


def set_device_scope():
    """
    设置当前程序可见的设备范围(当前程序只会使用自己可见的设备，不可见的设备不会被当前程序使用)
    1. tf.config.set_visible_devices
    2. CUDA_VISIBLE_DEVICES 也可以控制程序所使用的 GPU
        $ export CUDA_VISIBLE_DEVICES=2,3
    """
    # tf.config.set_visible_devices
    gpus = tf.config.list_physical_devices(device_type = "GPU")
    tf.config.set_visible_devices(devices = gpus[0:2], device_type = "GPU")
    # CUDA_VISIBLE_DEVICES 也可以控制程序所使用的 GPU
    os.environ("CUDA_VISIBLE_DEVICES") = "1,2,3"


def gpu_memory_strategy():
    """
    设置显存使用策略
    1.仅在需要时申请显存空间
    2.限制消耗固定大小的显存
    """
    # 仅在需要时申请显存空间
    gpus = tf.config.list_physical_devices(device_type = "GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device = gpu, enable = True)

    # 限制消耗固定大小的显存为 1GB
    gpus = tf.config.list_physical_devices(device_type = "GPU")
    tf.config.set_virtual_device_configuration(
        gpus[0], 
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 1024)]
    )


def one_to_muli_gpu():
    # 单 GPU 模拟多 GPU 环境
    # 在实体 GPU (GPU:0) 的基础上建立两个显存均为2GB的虚拟 GPU
    gpus = tf.config.list_physical_devices(device_type = "GPU")
    tf.config.set_virtual_device_configuration(
        gpus[0],
        [
            tf.config.VirtualDeviceConfiguration(memory_limit = 2048), 
            tf.config.VirtualDeviceConfiguration(memory_limit = 2048),
        ]
    )




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
