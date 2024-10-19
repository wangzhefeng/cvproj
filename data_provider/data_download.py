# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_download.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091417
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import requests
from urllib.parse import unquote
import skimage
from PIL import Image, ImageOps

import torch
from torchvision.transforms.functional import (
    to_pil_image, 
    pil_to_tensor,
)

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def skimage_url_image(url):
    """
    scikit-image 加载网络图片
    """
    arr = skimage.io.imread(url)

    return Image.fromarray(arr)


def pillow_url_image(url):
    """
    pillow 加载网络图片
    """
    image = Image.open(requests.get(url, stream = True).raw)
    image = ImageOps.exif_transpose(image)

    return image


def github_file(url, save_name = None):
    """
    下载 github 上的图片
    """
    raw_url = url \
        .replace('://github.com/', '://raw.githubusercontent.com/') \
        .replace('/blob/', '/')
    
    if save_name is None:
        save_name = unquote(os.path.basename(raw_url))
    
    torch.hub.download_url_to_file(raw_url, save_name)
    print('saved file: ' + save_name, file = sys.stderr)

    return save_name


def image_download(img_url: str, img_name: str, img_path: str = "./data/cv_clf_imgs/"):
    """
    图像下载，转换为 Tensor
    """
    # image download
    exists_status = os.path.exists(os.path.join(img_path, img_name))
    # print(exists_status)
    if not exists_status:
        os.system(f"wget {img_url} ; mv {img_name} {img_path}")
    # load image
    img = Image.open(f"{img_path}{img_name}")
    # image to tensor
    img_int = pil_to_tensor(img)
    
    return img, img_int



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
