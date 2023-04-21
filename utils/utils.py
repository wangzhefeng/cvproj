# -*- coding: utf-8 -*-


# ***************************************************
# * File        : test.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-20
# * Version     : 0.1.042023
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import argparse
import random
import datetime
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
import torch

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def seed_everything(seed = 42):
    """
    TODO

    Args:
        seed (int, optional): _description_. Defaults to 42.

    Returns:
        _type_: _description_
    """
    print(f"Global seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


def colorful(obj, color = "red", display_type = "plain"):
    """
    TODO
    # =======================================
    # 彩色输出格式：
    # -----------
    # 设置颜色开始 ：\033[显示方式;前景色;背景色m
    # ---------------------------------------
    # 说明：
    # -----------
    # 前景色            背景色           颜色
    # ---------------------------------------
    # 30                40              黑色
    # 31                41              红色
    # 32                42              绿色
    # 33                43              黃色
    # 34                44              蓝色
    # 35                45              紫红色
    # 36                46              青蓝色
    # 37                47              白色
    # ---------------------------------------
    # 显示方式           意义
    # ---------------------------------------
    # 0                终端默认设置
    # 1                高亮显示
    # 4                使用下划线
    # 5                闪烁
    # 7                反白显示
    # 8                不可见
    # =======================================
    Args:
        obj (_type_): _description_
        color (str, optional): _description_. Defaults to "red".
        display_type (str, optional): _description_. Defaults to "plain".

    Returns:
        _type_: _description_
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
    s = str(obj)
    color_code = color_dict.get(color, "")
    display  = display_type_dict.get(display_type, "")
    out = '\033[{};{}m'.format(display, color_code) + s + '\033[0m'

    return out


def prettydf(df, nrows = 20, ncols = 20, show = True):
    """
    TODO

    Args:
        df (_type_): _description_
        nrows (int, optional): _description_. Defaults to 20.
        ncols (int, optional): _description_. Defaults to 20.
        show (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    from prettytable import PrettyTable
    if len(df) > nrows:
        df = df.head(nrows).copy()
        df.loc[len(df)] = '...'
    
    if len(df.columns) > ncols:
        df = df.iloc[:, :ncols].copy()
        df['...'] = '...'
     
    def fmt(x):
        if isinstance(x, (float, np.float64)):
            return str(round(x, 5))
        else:
            s = str(x) if len(str(x)) < 9 else str(x)[:6] + '...'
            for char in ['\n', '\r', '\t', '\v', '\b']:
                s = s.replace(char, ' ')
            return s
        
    df = df.applymap(fmt)
    table = PrettyTable()
    table.field_names = df.columns.tolist()
    rows =  df.values.tolist()
    table.add_rows(rows)
    if show:
        print(table)
    
    return table


def namespace2dict(namespace):
    """
    TODO

    Args:
        namespace (_type_): _description_

    Returns:
        _type_: _description_
    """
    result = {}
    for k, v in vars(namespace).items():
        if not isinstance(v, Namespace):
            result[k] = v
        else:
            v_dic = namespace2dict(v)
            for v_key,v_value in v_dic.items():
                result[k + "." + v_key] = v_value

    return result 


def get_call_file(): 
    """
    TODO

    Returns:
        _type_: _description_
    """
    import traceback
    stack = traceback.extract_stack()

    return stack[-2].filename 


def getNotebookPath():
    """
    TODO

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    from jupyter_server import serverapp
    from jupyter_server.utils import url_path_join
    import requests,re
    kernelIdRegex = re.compile(r"(?<=kernel-)[\w\d\-]+(?=\.json)")
    kernelId = kernelIdRegex.search(get_ipython().config["IPKernelApp"]["connection_file"])[0]
    for jupServ in serverapp.list_running_servers():
        for session in requests.get(url_path_join(jupServ["url"], "api/sessions"), params={"token":jupServ["token"]}).json():
            if kernelId == session["kernel"]["id"]:
                return str(Path(jupServ["root_dir"]) / session["notebook"]['path']) 
    raise Exception('failed to get current notebook path')




# 测试代码 main 函数
def main():
    a = "test"
    out = colorful(a)
    print(out)

if __name__ == "__main__":
    main()
