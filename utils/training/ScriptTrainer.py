# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ScriptTrainer.py
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
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from utils_log import printlog
from utils.metrics import Accuracy

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# epoch
epochs = 20
# model save
ckpt_path = 'checkpoint.pt'
# early_stopping
monitor = "val_acc"
patience = 5
mode = "max"

# ------------------------------
# TODO
# ------------------------------
dl_train = None
dl_val = None
model = None
loss_fn = None
optimizer = None
metrics_dict = {
    "acc": Accuracy(task = "binary")
}


# 训练循环
history = {}
for epoch in range(1, epochs + 1):
    printlog(f"Epoch {epoch} / {epochs}")
    # ------------------------------
    # training
    # ------------------------------
    model.train()
    # --------------
    # training epoch
    # --------------
    total_loss, step = 0, 0
    loop = tqdm(enumerate(dl_train), total = len(dl_train))
    train_metrics_dict = deepcopy(metrics_dict)
    for batch_idx, batch in loop:
        # -----------------
        # training batch step
        # -----------------
        # batch dataset
        features, labels = batch
        # forward
        preds = model(features)
        loss = loss_fn(preds, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # training metrics
        step_metrics = {
            "train_" + name: metric_fn(preds, labels).item() 
            for name, metric_fn in train_metrics_dict.items()
        }
        step_log = dict({"train_loss": loss.item()}, **step_metrics)
        # 总损失和训练迭代次数更新
        total_loss += loss.item()
        step += 1
        if batch_idx != len(dl_train) - 1:
            loop.set_postfix(**step_log)
        else:
            epoch_loss = total_loss / step
            epoch_metrics = {
                "train_" + name: metric_fn.compute().item() 
                for name, metric_fn in train_metrics_dict.items()
            }
            epoch_log = dict({"train_loss": epoch_loss}, **epoch_metrics)
            loop.set_postfix(**epoch_log)
            # 重置 metric_fn
            for name, metric_fn in train_metrics_dict.items():
                metric_fn.reset()
    # result collect
    for name, metric in epoch_log.items():
        history[name] = history.get(name, []) + [metric]
    # ------------------------------
    # validate
    # ------------------------------
    model.eval()
    # --------------
    # validate epoch
    # --------------
    total_loss, step = 0, 0
    loop = tqdm(enumerate(dl_val), total = len(dl_val))
    val_metrics_dict = deepcopy(metrics_dict)
    with torch.no_grad():
        for batch_idx, batch in loop:
            # batch dataset
            features, labels = batch            
            # forward
            preds = model(features)
            loss = loss_fn(preds, labels)
            # validate metrics
            step_metrics = {
                "val_" + name: metric_fn(preds, labels).item() 
                for name,metric_fn in val_metrics_dict.items()
            }
            step_log = dict({"val_loss": loss.item()}, **step_metrics)
            # 总损失和训练迭代次数更新
            total_loss += loss.item()
            step += 1
            if batch_idx != len(dl_val) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {
                    "val_" + name: metric_fn.compute().item()
                    for name, metric_fn in val_metrics_dict.items()
                }
                epoch_log = dict({"val_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)
                # 重置 metric_fn
                for name, metric_fn in val_metrics_dict.items():
                    metric_fn.reset()
    # result collect
    epoch_log["epoch"] = epoch           
    for name, metric in epoch_log.items():
        history[name] = history.get(name, []) + [metric]
    # ------------------------------
    # early-stopping
    # ------------------------------
    arr_scores = history[monitor]
    best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
    if best_score_idx == len(arr_scores) - 1:
        torch.save(model.state_dict(), ckpt_path)
        print(f"<<<<<< reach best {monitor} : {arr_scores[best_score_idx]} >>>>>>", file = sys.stderr)
    # 在一定 epoch 内分数没有改进，停止训练
    if len(arr_scores) - best_score_idx > patience:
        print(f"<<<<<< {monitor} without improvement in {patience} epoch, early stopping >>>>>>", file = sys.stderr)
        break
    # 加载模型 checkpoint
    model.load_state_dict(torch.load(ckpt_path))

# 模型训练结果
df_history = pd.DataFrame(history)
print(df_history)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
