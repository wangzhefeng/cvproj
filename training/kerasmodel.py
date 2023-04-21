# -*- coding: utf-8 -*-


# ***************************************************
# * File        : kerasmodel.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-04
# * Version     : 0.1.040419
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
from tqdm import tqdm
from copy import deepcopy

import numpy as np 
import pandas as pd
import torch

from utils.printlog import printlog


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class StepRunner:

    def __init__(self,
                 net, 
                 loss_fn,
                 stage = "train",
                 metrics_dict = None,
                 optimizer = None,
                 lr_scheduler = None,
                 accelerator = None):
        self.net = net
        self.loss_fn = loss_fn
        self.metrics_dict = metrics_dict
        self.stage = stage
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

    def setp(self, features, labels):
        # forward
        preds = self.net(features)
        loss = self.loss_fn(preds, labels)
        # backward
        if self.optimizer is not None and self.stage == "train":
            # backward
            if self.accelerator is None:
                loss.backward()
            else:
                self.accelerator.backward(loss)
            # optimizer
            self.optimizer.step()
            # learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            # zero grad
            self.optimizer.zero_grad()
        # metrics
        step_metrics = {
            self.stage + "_" + name: metric_fn(preds, labels).item() 
            for name, metric_fn in self.metrics_dict.items()
        }
        return loss.item(), step_metrics

    def train_step(self, features, labels):
        """
        训练模式，dropout 层发生作用
        """
        self.net.train()
        return self.step(features, labels)

    @torch.no_grad()
    def eval_step(self, features, labels):
        """
        预测模式，dropout 层不发生作用
        """
        self.net.eval()
        return self.step(features, labels)

    def __call__(self, features, labels):
        if self.stage == "train":
            return self.train_step(features, labels)
        else:
            return self.eval_step(features, labels)


class EpochRunner:

    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.steprunner.net.train() if self.stage == "train" else self.steprunner.net.eval()

    def __call__(self, dataloader):
        total_loss = 0
        step = 0
        loop = tqdm(enumerate(dataloader), total = len(dataloader))
        for i, batch in loop:
            # step runner
            if self.stage == "train":
                loss, step_metrics = self.steprunner(*batch)
            else:
                with torch.no_grad():
                    loss, step_metrics = self.steprunner(*batch)
            step_log = dict({self.stage + "_loss": loss}, **step_metrics)
            # metrics
            total_loss += loss
            step += 1
            if i != len(dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {
                    self.stage + "_" + name: metric_fn.compute().item()
                    for name, metric_fn in self.steprunner.metrics_dict.items()
                }
                epoch_log = dict({self.stage + "_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log


def train_model(net, optimizer, loss_fn, metrics_dict, 
                train_data, val_data = None, 
                epochs = 10, 
                ckpt_path = 'checkpoint.pt',
                patience = 5, 
                monitor = "val_loss", 
                mode = "min"):
    history = {}
    for epoch in range(1, epochs + 1):
        printlog(f"Epoch {epoch} / {epochs}")
        # ------------------------------
        # train
        # ------------------------------
        train_step_runner = StepRunner(
            net = net,
            stage = "train",
            loss_fn = loss_fn,
            metrics_dict = deepcopy(metrics_dict),
            optimizer = optimizer,
        )
        train_epoch_runner = EpochRunner(train_step_runner)
        train_metrics = train_epoch_runner(train_data)
        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]
        # ------------------------------
        # validate
        # ------------------------------
        if val_data:
            val_step_runner = StepRunner(
                net = net,
                stage = "val",
                loss_fn = loss_fn,
                metrics_dict = deepcopy(metrics_dict),
            )
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_data)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]
        # ------------------------------
        # early-stopping
        # ------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            torch.save(net.state_dict(), ckpt_path)
            print(f"<<<<<< reach best {monitor} : {arr_scores[best_score_idx]} >>>>>>", file = sys.stderr)
        
        if len(arr_scores) - best_score_idx > patience:
            print(f"<<<<<< {monitor} without improvement in {patience} epoch, early stopping >>>>>>", file = sys.stderr)
            break 

        net.load_state_dict(torch.load(ckpt_path))

    return pd.DataFrame(history)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
