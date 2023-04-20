# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_load.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-24
# * Version     : 0.1.022421
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # or any {"0", "1", "2"}
import tensorflow as tf
from config.config_loader import settings


def data_loader():
    """
    When working with lots of real-world image data, 
    corrupted images are a common occurence. 
    Let's filter out badly-encoded images that do not 
    feature the string "JFIF" in their header.
    """
    num_skipped = 0
    for folder_name in set(settings["PATH"]["folder_names"]):
        folder_path = os.path.join(settings["PATH"]["data_root_path"], folder_name)
        for fname in os.listdir(folder_path):
            file_path = os.path.join(folder_path, fname)
            try:
                fobj = open(file_path, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()
            
            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(file_path)

    print("Deleted %d images" % num_skipped)




def main():
    # 1.去除异常格式的图片数据
    data_loader()


if __name__ == "__main__":
    main()

