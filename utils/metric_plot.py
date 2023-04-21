# -*- coding: utf-8 -*-


# ***************************************************
# * File        : plots.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-21
# * Version     : 0.1.042100
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def plot_metric(dfhistory, metric):
    """
    TODO

    Args:
        dfhistory (_type_): _description_
        metric (_type_): _description_

    Returns:
        _type_: _description_
    """
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




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
