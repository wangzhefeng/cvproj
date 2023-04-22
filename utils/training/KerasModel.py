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
import sys
import datetime
from tqdm import tqdm
from copy import deepcopy

import numpy as np 
import pandas as pd
import torch
from torch import nn
from accelerate import Accelerator

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def colorful(obj, color = "red", display_type = "plain"):
    """
    =======================================
    彩色输出格式：
    -----------
    设置颜色开始 ："\033[显示方式;前景色;背景色m"
    ---------------------------------------
    说明：
    -----------
    前景色            背景色           颜色
    ---------------------------------------
    30                40              黑色
    31                41              红色
    32                42              绿色
    33                43              黃色
    34                44              蓝色
    35                45              紫红色
    36                46              青蓝色
    37                47              白色
    ---------------------------------------
    显示方式           意义
    ---------------------------------------
    0                终端默认设置
    1                高亮显示
    4                使用下划线
    5                闪烁
    7                反白显示
    8                不可见
    =======================================
    """
    color_dict = {
        "black": "30", 
        "red": "31", 
        "green": "32", 
        "yellow": "33",
        "blue": "34", 
        "purple": "35",
        "cyan":"36", 
        "white":"37"
    }
    display_type_dict = {
        "plain": "0",
        "highlight": "1",
        "underline": "4",
        "shine": "5",
        "inverse": "7",
        "invisible": "8"
    }
    color_code = color_dict.get(color, "")
    display_code  = display_type_dict.get(display_type, "")
    out = f"\033[{display_code};{color_code}m" + str(obj) + "\033[0m"

    return out


class StepRunner:

    def __init__(self, net, loss_fn, stage = "train", metrics_dict = None, optimizer = None, lr_scheduler = None, accelerator = None):
        self.net = net
        self.loss_fn = loss_fn
        self.stage = stage
        self.metrics_dict = metrics_dict
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

    def __call__(self, batch):
        # batch dataset
        features, labels = batch
        # forward
        preds = self.net(features)
        loss = self.loss_fn(preds, labels)
        # backward
        if self.optimizer is not None and self.stage == "train":
            self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
        # accelerator
        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)
        all_loss = self.accelerator.gather(loss).sum()
        # losses
        step_losses = {
            self.stage + "_loss": all_loss.item()
        }
        # metrics
        step_metrics = {
            self.stage + "_" + name: metric_fn(all_preds, all_labels)
            for name, metric_fn in self.metrics_dict.items()
        }
        if self.stage == "train":
            if self.optimizer is not None:
                step_metrics["lr"] = self.optimizer.state_dict()["param_groups"][0]["lr"]
            else:
                step_metrics["lr"] = 0.0
        return step_losses, step_metrics


class EpochRunner:

    def __init__(self, steprunner, quiet = False):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.steprunner.net.train() if self.stage == "train" else self.steprunner.net.eval()
        self.accelerator = self.steprunner.accelerator
        self.quiet = quiet

    def __call__(self, dataloader):
        loop = tqdm(
            enumerate(dataloader, start = 1), 
            total = len(dataloader),
            file = sys.stdout,
            disable = not self.accelerator.is_local_main_process or self.quiet,
            ncols = 100,
        )
        epoch_losses = {}
        for step, batch in loop:
            # step runner
            if self.stage == "train":
                step_losses, step_metrics = self.steprunner(batch)
            else:
                with torch.no_grad():
                    step_losses, step_metrics = self.steprunner(batch)
            step_log = dict(step_losses, **step_metrics)
            # loss collect
            for key, value in step_losses.items():
                epoch_losses[key] = epoch_losses.get(key, 0.0) + value
            # metrics
            if step != len(dataloader):
                loop.set_postfix(**step_log)
            else:
                epoch_metrics = step_metrics
                epoch_metrics.update({
                    self.stage + "_" + name: metric_fn.compute().item()
                    for name, metric_fn in self.steprunner.metrics_dict.items()
                })
                epoch_losses = {
                    key: value / step 
                    for key, value in epoch_losses.items()
                }
                epoch_log = dict(epoch_losses, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log


class KerasModel(nn.Module):

    StepRunner, EpochRunner = StepRunner, EpochRunner

    def __init__(self, net, loss_fn, metrics_dict = None, optimizer = None, lr_scheduler = None) -> None:
        super(KerasModel).__init__()
        self.net = net
        self.loss_fn = loss_fn
        self.metrics_dict = torch.nn.ModuleDict(metrics_dict)
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(self.net.parameters(), lr = 1e-3)
        self.lr_scheduler = lr_scheduler
        self.from_scratch = True

    def load_ckpt(self, ckpt_path = "checkpoint.pt"):
        self.net.load_state_dict(torch.load(ckpt_path))
        self.from_scratch = False

    def forward(self, x):
        return self.net.forward(x)

    def fit(self, train_data, val_data = None, epochs = 10, ckpt_path = "checkpoint.pt", 
            patience = 5, monitor = "val_loss", mode = "min", mixed_precision = "no",
            callbacks = None, plot = False, quiet = False):
        self.__dict__.update(locals())
        # accelerator
        self.accelerator = Accelerator(mixed_precision = mixed_precision)

        # device
        device = str(self.accelerator.device)
        device_type = "🐌" if "cpu" in device else "⚡️"
        self.accelerator.print(colorful(f"<<<<<< {device_type} {device} is used >>>>>"))

        # params
        self.net, self.loss_fn, self.metrics_dict, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.net, 
            self.loss_fn, 
            self.metrics_dict,
            self.optimizer, 
            self.lr_scheduler,
        )

        # dataloader
        train_dataloader, val_dataloader = self.accelerator.prepare(train_data, val_data)

        # callbacks
        callbacks = callbacks if callbacks is not None else []
        # plot callback
        if plot == True:
            from Callbacks.VisProgress import VisProgress
            callbacks.append(VisProgress(self))
        self.callbacks = self.accelerator.prepare(callbacks) 
        if self.accelerator.is_local_main_process:
            for callback_obj in self.callbacks:
                callback_obj.on_fit_start(model = self)
        # ------------------------------
        # model training and validation
        # ------------------------------
        self.history = {}
        start_epoch = 1 if self.from_scratch else 0
        for epoch in range(start_epoch, epochs + 1):
            # log
            should_quiet = False if quiet == False else (quiet == True or epoch > quiet)
            if not should_quiet:
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.accelerator.print("\n" + "=" * 80 + str(nowtime))
            # ------------------------------
            # training
            # ------------------------------
            # model training
            train_step_runner = self.StepRunner(
                net = self.net,
                loss_fn = self.loss_fn,
                stage = "train",
                metrics_dict = deepcopy(self.metrics_dict),
                optimizer = self.optimizer if epoch > 0 else None,
                lr_scheduler = self.lr_scheduler if epoch > 0 else None,
                accelerator = self.accelerator,
            )
            train_epoch_runner = self.EpochRunner(steprunner = train_step_runner, quiet = should_quiet)
            train_epoch_metric = train_epoch_runner(dataloader = train_dataloader)
            # training metric
            train_metrics = {"epoch": epoch}
            train_metrics.update(train_epoch_metric)
            for name, metric in train_metrics.items():
                self.history[name] = self.history.get(name, []) + [metric]
            
            if self.accelerator.is_local_main_process:
                for callback_obj in self.callbacks:
                    callback_obj.on_train_epoch_end(model = self)
            # ------------------------------
            # validate
            # ------------------------------
            if val_dataloader:
                # model validation
                val_step_runner = self.StepRunner(
                    net = self.net,
                    loss_fn = self.loss_fn,
                    stage = "val",
                    metrics_dict = deepcopy(self.metrics_dict),
                    accelerator = self.accelerator,
                )
                val_epoch_runner = self.EpochRunner(
                    steprunner = val_step_runner, 
                    quiet = should_quiet,
                )
                with torch.no_grad():
                    val_metrics = val_epoch_runner(dataloader = val_dataloader)
                # validation metric
                for name, metric in val_metrics.items():
                    self.history[name] = self.history.get(name, []) + [metric]
                
                if self.accelerator.is_local_main_process:
                    for callback_obj in self.callbacks:
                        callback_obj.on_validation_epoch_end(model = self)
            # ------------------------------
            # early-stopping
            # ------------------------------
            self.accelerator.wait_for_everyone()
            arr_scores = self.history[monitor]
            best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)

            if best_score_idx == len(arr_scores) - 1:
                net_dict = self.accelerator.get_state_dict(self.net)
                self.accelerator.save(net_dict, ckpt_path)
                if not should_quiet:
                    self.accelerator.print(colorful(f"<<<<<< reach best {monitor} : {arr_scores[best_score_idx]} >>>>>>"))
            
            if len(arr_scores) - best_score_idx > patience:
                self.accelerator.print(colorful(f"<<<<<< {monitor} without improvement in {patience} epoch, early stopping >>>>>>"))
                break;
        # ------------------------------
        # result collect
        # ------------------------------
        if self.accelerator.is_local_main_process:
            # model history
            df_history = pd.DataFrame(self.history)
            self.accelerator.print(df_history)
            # callback
            for callback_obj in self.callbacks:
                callback_obj.on_fit_end(model = self)
            
            self.net = self.accelerator.unwrap_model(self.net)
            self.net.load_state_dict(torch.load(ckpt_path))
            return df_history

    @torch.no_grad()
    def evaluate(self, val_data):
        # accelerator
        accelerator = Accelerator()
        # params
        self.net, self.loss_fn, self.metrics_dict = accelerator.prepare(
            self.net, 
            self.loss_fn, 
            self.metrics_dict
        )
        # data
        val_data = accelerator.prepare(val_data)
        # validation
        val_step_runner = self.StepRunner(
            net = self.net, 
            loss_fn = self.loss_fn,
            stage = "val",
            metrics_dict = deepcopy(self.metrics_dict),
            accelerator = accelerator,
        )
        val_epoch_runner = self.EpochRunner(steprunner = val_step_runner)
        val_metrics = val_epoch_runner(dataloader = val_data)
        return val_metrics




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
