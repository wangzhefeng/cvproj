# -*- coding: utf-8 -*-

# ***************************************************
# * File        : MiniLogCallback.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-22
# * Version     : 0.1.042215
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import sys
import datetime

import pandas as pd

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class MiniLogCallback:
    
    def __init__(self) -> None:
        pass

    def on_fit_start(self, model: "KerasModel"):
        pass

    def on_train_epoch_end(self, model: "KerasModel"):
        pass

    def on_validation_epoch_end(self, model: "KerasModel"):
        # model history
        df_history = pd.DataFrame(model.history)
        # model epoch
        epoch = max(df_history["epoch"])
        # model monitor metrics
        monitor_arr = df_history[model.monitor]
        # model best score
        best_monitor_score = monitor_arr.max() if model.name == "max" else monitor_arr.min()
        # log
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(
            f"epoch 【{epoch}】@{nowtime} --> best_{model.monitor} = {str(best_monitor_score)}",
            file = sys.stderr,
            end = "\r"
        )

    def on_fit_end(self, model: "KerasModel"):
        pass




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
