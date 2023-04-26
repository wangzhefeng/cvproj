# -*- coding: utf-8 -*-

# ***************************************************
# * File        : metric_plot.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-21
# * Version     : 0.1.042100
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import matplotlib.pyplot as plt

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def plot_importance(features, importances, topk = 20):
    """
    特征重要性绘图

    Args:
        features (_type_): _description_
        importances (_type_): _description_
        topk (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """
    import pandas as pd
    import plotly.express as px 
    # feature importance
    dfimportance = pd.DataFrame({
        'feature': features,
        'importance': importances
    })
    dfimportance = dfimportance.sort_values(by = "importance").iloc[-topk:]
    fig = px.bar(
        dfimportance, 
        x = "importance", 
        y = "feature", 
        title = "Feature Importance"
    )

    return fig
 

def plot_score_distribution(labels, scores):
    """
    for binary classification problem.

    Args:
        labels (_type_): _description_
        scores (_type_): _description_

    Returns:
        _type_: _description_
    """
    import plotly.express as px 
    fig = px.histogram(
        x = scores, 
        color = labels,  
        nbins = 50,
        title = "Score Distribution",
        labels = dict(color = 'True Labels', x = 'Score')
    )

    return fig


def plot_metrics(history_df, metric):
    train_metrics = history_df["train_" + metric]
    val_metrics = history_df["val_" + metric]
    epochs = range(1, len(train_metrics) + 1)
    
    plt.plot(epochs, train_metrics, "bo--")
    plt.plot(epochs, val_metrics, "ro--")
    plt.title("Training and Validation " + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, "val_" + metric])
    plt.show()


def plot_linearreg_metrics(X, Y, preds, feature_idx, subplot_idx, xlabel, ylabel):
    plt.figure(figsize = (8, 8))
    ax = plt.subplot(subplot_idx)
    ax.scatter(
        X[:, feature_idx].numpy(), 
        Y[:, 0].numpy(), 
        c = "b", 
        label = "samples"
    )
    ax.plot(
        X[:, feature_idx].numpy(), 
        preds, 
        c = "-r", 
        linewidth = 5.0, 
        label = "model"
    )
    ax.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel, rotation = 0)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
