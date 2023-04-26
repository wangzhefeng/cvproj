# -*- coding: utf-8 -*-

# ***************************************************
# * File        : merge_dataset_folders.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-25
# * Version     : 0.1.042519
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import shutil
from pathlib import Path

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def merge_dataset_folders(from_folders, to_folder, rename_file = True):
    """
    _summary_

    Args:
        from_folders (_type_): _description_
        to_folder (_type_): _description_
        rename_file (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    get_file_num = lambda x: len([y for y in Path(x).rglob('*') if y.is_file()])
    print('before merge:')
    for x in from_folders:
        print(f'{x}: {get_file_num(x)} files')

    done_files = set()
    for i,folder in enumerate(from_folders):
        shutil.copytree(folder, to_folder, dirs_exist_ok = True)
        folder_name = Path(folder).name
        if rename_file:
            files = {x.absolute() for x in Path(to_folder).rglob('*') if x.is_file()}
            todo_files = files - done_files
            for x in todo_files:
                new_name = folder_name + '_' + str(i) + '_' + x.name
                y = x.rename(x.parent / new_name)
                done_files.add(y)
    print('\nafter merge:')
    print(f'{to_folder}: {get_file_num(to_folder)} files')

    return to_folder



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
