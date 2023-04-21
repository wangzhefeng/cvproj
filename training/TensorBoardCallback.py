# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lightcallbacks.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-21
# * Version     : 0.1.042116
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
import datetime
import argparse
from copy import deepcopy

import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import PIL
# import lightning.pytorch as pl
import pytorch_lightning as pl

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def namespace2dict(namespace):
    result = {}
    for k, v in vars(namespace).items():
        if not isinstance(v, argparse.Namespace):
            result[k] = v
        else:
            v_dic = namespace2dict(v)
            for v_key, v_value in v_dic.items():
                result[k + "." + v_key] = v_value
    return result 


class TensorBoardCallback(pl.callbacks.Callback):

    def __init__(
        self,
        save_dir="tb_logs",
        model_name="default",
        log_weight=True,
        log_weight_freq=5,
        log_graph=True,
        example_input_array=None,
        log_hparams=True,
        hparams_dict=None
        ) -> None:
        super.__init__()
        self.logger = pl.loggers.TensorBoardLogger(save_dir, model_name)
        self.writer = self.logger.experiment
        self.log_graph = log_graph
        self.log_weight = log_weight
        self.log_weight_freq = log_weight_freq
        self.example_input_array = example_input_array
        self.log_hparams = log_hparams
        self.hparams_dict = namespace2dict(hparams_dict) if isinstance(hparams_dict, Namespace) else hparams_dict
    
    # ------------------------------
    # 
    # ------------------------------
    def on_fit_start(self, trainer, pl_module):
        pass

    def on_fit_end(self, trainer, pl_module):
        pass

    # ------------------------------
    # 
    # ------------------------------
    def on_before_backward(self, trainer, pl_module, loss):
        pass

    def on_after_backward(self, trainer, pl_module):
        pass

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        pass
    
    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        pass
    
    def on_sanity_check_start(self, trainer, pl_module):
        pass

    def on_sanity_check_end(self, trainer, pl_module):
        pass

    # ------------------------------
    # train
    # ------------------------------
    def on_train_start(self, trainer, pl_module):
        pass

    def on_train_epoch_start(self, trainer, pl_module):
        pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pass

    def on_train_epoch_end(self, trainer, pl_module):
        pass
    
    def on_train_end(self, trainer, pl_module):
        pass
    
    # ------------------------------
    # validation
    # ------------------------------
    def on_validation_start(self, trainer, pl_module):
        pass

    def on_validation_epoch_start(self, trainer, pl_module):
        pass

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_validation_end(self, trainer, pl_module):
        pass

    # ------------------------------
    # test
    # ------------------------------
    def on_test_start(self, trainer, pl_module):
        pass
    
    def on_test_end(self, trainer, pl_module):
        pass

    def on_test_epoch_start(self, trainer, pl_module):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        pass

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        pass
    
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pass

    # ------------------------------
    # prediction
    # ------------------------------
    def on_predict_start(self, trainer, pl_module):
        pass

    def on_predict_epoch_start(self, trainer, pl_module):
        pass

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx):
        pass

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pass
    
    def on_predict_epoch_end(self, trainer, pl_module):
        pass
    
    def on_predict_end(self, trainer, pl_module):
        pass

    # ------------------------------
    # checkpoint
    # ------------------------------
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        pass

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        pass

    # ------------------------------
    # exception
    # ------------------------------
    def on_exception(self, trainer, pl_module, exception):
        pass
    

    




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
