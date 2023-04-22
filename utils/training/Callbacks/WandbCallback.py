# -*- coding: utf-8 -*-

# ***************************************************
# * File        : WandbCallback.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-22
# * Version     : 0.1.042215
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import argparse
import datetime

import pandas as pd
import wandb

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def plot_metric(dfhistory, metric):
    import plotly.graph_objs as go
    # metric
    train_metrics = dfhistory["train_" + metric].values.tolist()
    val_metrics = dfhistory['val_' + metric].values.tolist()
    # epochs
    epochs = list(range(1, len(train_metrics) + 1))
    # train
    train_scatter = go.Scatter(
        x = epochs, 
        y = train_metrics, 
        mode = "lines+markers",
        name = 'train_' + metric, 
        marker = dict(size = 8, color = "blue"),
        line = dict(width = 2, color = "blue", dash = "dash")
    )
    # validation
    val_scatter = go.Scatter(
        x = epochs, 
        y = val_metrics, 
        mode = "lines+markers",
        name = 'val_' + metric,
        marker = dict(size = 10, color = "red"),
        line = dict(width = 2, color = "red", dash = "solid")
    )
    fig = go.Figure(data = [train_scatter, val_scatter])

    return fig


class WandbCallback:
    
    def __init__(self, 
                 project = None, 
                 config = None, 
                 name = None, 
                 save_ckpt = True, 
                 save_code = True) -> None:
        self.__dict__.update(locals())
        self.project = project
        if isinstance(config, argparse.Namespace):
            self.config = config.__dict__
        if name is None:
            self.name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save_ckpt = save_ckpt
        self.save_code = save_code
    
    def on_fit_start(self, model: "KerasModel"):
        if wandb.run is None:
            wandb.init(
                project = self.project, 
                config = self.config, 
                name = self.name, 
                save_code = self.save_code
            )
        model.run_id = wandb.run.id
            
    def on_train_epoch_end(self, model: "KerasModel"):
        pass

    def on_validation_epoch_end(self, model: "KerasModel"):
        # model history
        df_history = pd.DataFrame(model.history)
        # define metrics
        n = len(df_history)
        if n == 1:
            for m in df_history.columns:
                wandb.define_metric(name = m, step_metric = "epoch", hidden = False if m != "epoch" else True)
            wandb.define_metric(name = "best_" + model.monitor, step_metric = "epoch")
        # metric monitor
        dic = dict(df_history.iloc[n - 1])
        monitor_arr = df_history[model.monitor]
        bset_monitor_score = monitor_arr.max() if model.mode == "max" else monitor_arr.min()
        dic.update({
            "best_" + model.monitor: bset_monitor_score,
        })
        wandb.run.summary["best_score"] = bset_monitor_score
        wandb.log(dic)

    def on_fit_end(self, model: "KerasModel"):
        # save df_history
        df_history = pd.DataFrame(model.history)
        df_history.to_csv(os.path.join(wandb.run.dir, "df_history.csv"), index = None)
        # save ckpt
        if self.save_ckpt:
            arti_model = wandb.Artifact("checkpoint", type = "model")
            arti_model.add_file(model.ckpt_path)
            wandb.log_artifact(arti_model)
        # plotly metrics
        metrics = [
            x.replace("train_", "").replace("val_", "") 
            for x in df_history.columns 
            if "train_" in x
        ]
        metrics_fig = {
            m + "_curve": plot_metric(df_history, m) for m in metrics
        }
        wandb.log(metrics_fig)
        run_dir = wandb.run.dir
        wandb.finish()
        # local save
        import shutil
        shutil.copy(
            model.ckpt_path, 
            os.path.join(run_dir, os.path.basename(model.ckpt_path)),
        )




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
