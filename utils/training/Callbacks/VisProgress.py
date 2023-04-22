# -*- coding: utf-8 -*-

# ***************************************************
# * File        : VisProgress.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-22
# * Version     : 0.1.042216
# * Description : description
# * Link        : https://github.com/fastai/fastprogress/
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import numpy as np
import pandas as pd

from fastprogress.fastprogress import master_bar, progress_bar

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class VisProgress:
    
    def __init__(self, figsize = (6, 4)) -> None:
        self.figsize = figsize

    def on_fit_start(self, model: "KerasModel"):
        # init master bar
        self.mb = master_bar(range(model.epochs))
        # metric
        self.metric = model.monitor.replace("val_", "")
        # model history
        df_history = pd.DataFrame(model.history)
        # master bar update
        self.mb.update_graph(
            df_history,
            self.metric,
            x_bounds = [0, min(10, model.epochs)],
            title = f"best {model.monitor} = ?", 
            figsize = self.figsize
        )
        self.mb.update(0)
        self.mb.show()

    def on_train_epoch_end(self):
        pass

    def get_title(self, model: "KerasModel"):
        # model history
        df_history = pd.DataFrame(model.history)
        # scores
        arr_scores = df_history[model.monitor]
        best_score = np.max(arr_scores) if model.mode == "max" else np.min(arr_scores)
        title = f"best {model.monitor} = {best_score:.4f}"
        return title

    def on_validation_epoch_end(self, model: "KerasModel"):
        df_history = pd.DataFrame(model.history)
        n = len(df_history)
        self.mb.update_graph(
            df_history,
            self.metric,
            x_bounds = [df_history["epoch"].min(), min(10 + (n // 10) * 10, model.epoch)],
            title = self.get_title(model),
            figsize = self.figsize,
        )
        self.mb.update(n)
        if n == 1:
            self.mb.write(df_history.columns, table = True)
        self.mb.write(df_history.iloc[n - 1], table = True)

    def on_fit_end(self, model: "KerasModel"):
        # model history
        df_history = pd.DataFrame(model.history)
        # master bar
        self.mb.update_graph(
            df_history,
            self.metric,
            title = self.get_title(model),
            figsize = self.figsize,
        )
        self.mb.on_iter_end()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
