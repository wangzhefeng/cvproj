# -*- coding: utf-8 -*-

# ***************************************************
# * File        : Ammonit.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-22
# * Version     : 0.1.042223
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from IPython import display
from matplotlib import pyplot as plt

from plt_config import use_svg_display, set_axes

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Animator:
    """
    For plotting data in animation.

    Example:
        animator = Animator(
            xlabel='epoch', 
            xlim=[1, num_epochs], ylim=[0.3, 0.9],
            legend=['train loss', 'train acc', 'test acc']
        )
    """
    def __init__(self, 
                 xlabel = None, ylabel = None, 
                 legend = None, 
                 xlim = None, ylim = None, 
                 xscale = 'linear', yscale = 'linear',
                 fmts = ('-', 'm--', 'g-.', 'r:'), 
                 nrows = 1, ncols = 1,
                 figsize = (3.5, 2.5)):
        # Incrementally plot multiple lines
        # legend
        if legend is None:
            legend = []
        
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize = figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], 
            xlabel, ylabel, 
            xlim, ylim, 
            xscale, yscale, 
            legend
        )
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait = True)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
