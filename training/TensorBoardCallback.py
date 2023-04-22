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
import datetime
import argparse
from copy import deepcopy

import torch
# from torch.utils.tensorboard import SummaryWriter
import lightning.pytorch as pl

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

    def __init__(self,
                 save_dir = "tb_logs",
                 model_name = "default",
                 log_weight = True,
                 log_weight_freq = 5,
                 log_graph = True,
                 example_input_array = None,
                 log_hparams = True,
                 hparams_dict = None) -> None:
        super.__init__()
        self.logger = pl.loggers.TensorBoardLogger(save_dir, model_name)
        self.writer = self.logger.experiment
        self.log_weight = log_weight
        self.log_weight_freq = log_weight_freq
        self.log_graph = log_graph
        self.example_input_array = example_input_array
        self.log_hparams = log_hparams
        self.hparams_dict = namespace2dict(hparams_dict) \
            if isinstance(hparams_dict, argparse.Namespace) \
            else hparams_dict
    # ------------------------------
    # model fit
    # ------------------------------
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when fit begins.

        Args:
            trainer (pl.Trainer): _description_
            pl_module (pl.LightningModule): _description_
        """
        # weight log
        # -----------
        if self.log_weight:
            for name, param in pl_module.named_parameters():
                self.writer.add_histogram(
                    name, 
                    param.clone().cpu().data.numpy(), 
                    -1
                )
            self.writer.flush()
        # graph log
        # -----------
        if not self.log_graph:
            return
        if self.example_input_array is None and pl_module.example_input_array is not None:
            self.example_input_array = pl_module.example_input_array
        if self.example_input_array is None:
            raise Exception("example_input_array needed for graph logging ...")
        net_cpu = deepcopy(pl_module.net).cpu()
        self.writer.add_graph(
            net_cpu, 
            input_to_model = [self.example_input_array]
        )
        self.writer.flush()
        # image log
        # -----------
        # from .plots import text2img, img2tensor
        # summary_text =  summary(net_cpu, input_data = self.example_input_array)
        # summary_tensor = img2tensor(text2img(summary_text))
        # self.writer.add_image('summary', summary_tensor, global_step = -1)
        # self.writer.flush()
        
        del(net_cpu)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Called when fit ends.

        Args:
            trainer (_type_): _description_
            pl_module (_type_): _description_
        """
        # weight log
        # -----------
        if self.log_weight:
            for name, param in pl_module.net.named_parameters():
                self.writer.add_histogram(
                    name, 
                    param.clone().cpu().data.numpy(), 
                    pl_module.current_epoch
                )
            self.writer.flush()
        # hparams log
        # -----------
        if self.log_hparams:
            hyper_dict = {
                "version": self.logger.version,
                "version_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            if self.hparams_dict is not None:
                hyper_dict.update(self.hparams_dict)
                for key, value in self.hparams_dict.items():
                    if not isinstance(value, (int, float, str, torch.Tensor)):
                        hyper_dict[key] = str(value)
            df_history = pl_module.get_history()
            monitor = trainer.checkpoint_callback.monitor
            mode = trainer.checkpoint_callback.mode
            best_idx = df_history[monitor].argmax() \
                if mode == "max" \
                else df_history[monitor].argmin()
            metric_dict = dict(
                df_history[[col for col in df_history.columns if col.startswith("val_")]].iloc[best_idx]
            )
            self.writer.add_hparams(hyper_dict, metric_dict)
            self.writer.flush()
        self.writer.close()
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

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = pl_module.current_epoch
        # metric log
        # -----------
        dic = deepcopy(pl_module.history[epoch])
        dic.pop("epoch", None)

        metrics_group = {}
        for key, value in dic.items():
            g = key.replace("train_", "").replace("val_", "")
            metrics_group[g] = dict(metrics_group.get(g, {}), **{key: value})

        for group, metrics in metrics_group.items():
            self.writer.add_scalars(group, metrics, epoch)
        self.writer.flush()
        # weight log
        # -----------
        if self.log_weight and (epoch + 1) % self.log_weight_freq == 0:
            for name, param in pl_module.net.named_parameters():
                self.writer.add_histogram(
                    name, 
                    param.clone().cpu().data.numpy(), 
                    epoch
                )
            self.writer.flush()
    
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

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = pl_module.current_epoch
        # metric log
        # -----------
        dic = deepcopy(pl_module.history[epoch])
        if epoch == 0 and "train_loss" not in dic:
            return
        dic.pop("epoch", None)

        metrics_group = {}
        for key, value in dic.items():
            g = key.replace("train_", "").replace("val_", "")
            metrics_group[g] = dict(metrics_group.get(g, {}), **{key: value})
        
        for group, metrics in metrics_group.items():
            self.writer.add_scalars(group, metrics, epoch) 
        self.writer.flush()

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
    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        """
        Called when any trainer execution is interrupted by an exception.
        """
        # weight log
        if self.log_weight:
            for name, param in pl_module.net.named_parameters():
                self.writer.add_histogram(
                    name, 
                    param.clone().cpu().data.numpy(), 
                    pl_module.current_epoch
                )
            self.writer.flush()
        self.writer.close() 




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
