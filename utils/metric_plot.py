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
import matplotlib.pyplot as plt

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def use_svg_display():
    """
    Use the svg format to display a plot in Jupyter.
    """
    from matplotlib_inline import backend_inline
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """
    Set the figure size for matplotlib.
    """
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """
    Set the axes for matplotlib.
    """
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y = None, 
         xlabel = None, ylabel = None, 
         legend = None, 
         xlim = None, ylim = None, 
         xscale = 'linear', yscale = 'linear',
         fmts = ('-', 'm--', 'g-.', 'r:'), 
         figsize = (3.5, 2.5), axes = None):
    """
    Plot data points.
    """
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # Return True if `X` (tensor or list) has 1 axis
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


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
