# -*- coding: utf-8 -*-

# ***************************************************
# * File        : lightmodel.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-21
# * Version     : 0.1.042119
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
import warnings
import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn
import lightning.pytorch as pl

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class LightningModel(pl.LightningModule):
    
    def __init__(self, 
                 net = None, 
                 loss_fn = None, 
                 metrics_dict = None, 
                 optimizer = None, 
                 lr_scheduler = None) -> None:
        super().__init__()
        self.net = net
        self.history = {}
        # metrics
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        # loss func
        self.loss_fn = loss_fn
        # optimizer
        self.optimizer = optimizer \
            if optimizer is not None \
            else torch.optim.Adam(self.parameters(), lr = 1e-2)
        # learning rate
        self.lr_scheduler = lr_scheduler 
        # parameters
        for p in ["net", "loss_fn", "metrics_dict", "optimizer", "lr_scheduler"]:
            self.save_hyperparameters(p)
    
    def forward(self, x):
        if self.net:
            return self.net.forward(x)
        else:
            raise NotImplementedError
    
    def backward(self, loss, optimizer, optimizer_idx):
        pass
    # ------------------------------
    # step
    # ------------------------------
    def shared_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)
        return {
            "y": y.detach(),
            "preds": preds.detach(),
            "loss": loss,
        }

    def shared_step_end(self, outputs, stage):
        if stage == "train":
            metrics = self.train_metrics    
        elif stage == "val":
            metrics = self.val_metrics
        elif stage == "test":
            metrics = self.test_metrics
        
        for name in metrics:
            step_metric = metrics[name](
                outputs["preds"], 
                outputs["y"]
            ).item()
            if stage == "train":
                self.log(name, step_metric, prob_bar = True)
        return outputs["loss"].mean()
    
    def shared_epoch_end(self, outputs, stage = "train"):
        if stage == "train":
            metrics = self.train_metrics    
        elif stage == "val":
            metrics = self.val_metrics
        elif stage == "test":
            metrics = self.test_metrics
            
        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(
            torch.tensor([
                t[(stage + "_loss").replace("train_", "")]
                for t in outputs
            ])
        ).item()
        dic = {
            "epoch": epoch, 
            stage + "_loss": stage_loss,
        }
        for name in metrics:
            epoch_metric = metrics[name].compute().item()
            metrics[name].reset()
            dic[stage + "_" + name] = epoch_metric
        
        if stage != "test":
            self.history[epoch] = dict(self.history.get(epoch, {}), **dic)
        return dic
    
    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return self.optimizer
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler,
        }
    # ------------------------------
    # training
    # ------------------------------
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def train_step_end(self, outputs):
        return {
            "loss": self.shared_step_end(outputs, "train")
        }

    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage = "train")
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger = True)
    # ------------------------------
    # validation
    # ------------------------------
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step_end(self, outputs):
        return {
            "val_loss": self.shared_epoch_end(outputs, "val")
        }
    
    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage = "val")
        self.print_bar()
        self.print(dic)
        dic.pop("epoch", None)
        self.log_dict(dic, logger = True)
        # log when reach best score
        ckpt_cb = self.trainer.checkpoint_callback
        monitor = ckpt_cb.monitor
        mode = ckpt_cb.mode
        arr_scores = self.get_history()[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            self.print(
                f"<<<<<< reach best {monitor} : {arr_scores[best_score_idx]}", 
                file = sys.stderr
            )
    # ------------------------------
    # test
    # ------------------------------
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def test_step_end(self, outputs):
        return {
            "test_loss": self.shared_epoch_end(outputs, "test")
        }
 
    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage = "test")
        dic.pop("epoch", None)
        self.print(dic)
        self.log_dict(dic, logger = True)
    # ------------------------------
    # predict
    # ------------------------------
    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 2:
            return self(batch[0])
        else:
            return self(batch)
    # ------------------------------
    # utils 
    # ------------------------------    
    def get_history(self):
        df_history = pd.DataFrame(self.history.values())
        return df_history

    def print_bar(self):
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n" + "=" * 80 + f"{nowtime}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
