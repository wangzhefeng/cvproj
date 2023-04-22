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
import PIL
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
    from IPython import get_ipython
    import requests
    import re
    kernelIdRegex = re.compile(r"(?<=kernel-)[\w\d\-]+(?=\.json)")
    kernelId = kernelIdRegex.search(get_ipython().config["IPKernelApp"]["connection_file"])[0]
    for jupServ in serverapp.list_running_servers():
        for session in requests.get(
            url_path_join(jupServ["url"], "api/sessions"), 
            params = {"token": jupServ["token"]}
        ).json():
            if kernelId == session["kernel"]["id"]:
                return str(Path(jupServ["root_dir"]) / session["notebook"]['path']) 
    raise Exception('failed to get current notebook path')




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
