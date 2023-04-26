# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_download.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-21
# * Version     : 0.1.042101
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from urllib.parse import unquote
from bs4 import BeautifulSoup

import requests
import skimage
import torch
from PIL import Image, ImageOps

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def skimage_url_image(url):
    arr = skimage.io.imread(url)

    return Image.fromarray(arr)


def pillow_url_image(url):
    image = Image.open(requests.get(url, stream = True).raw)
    image = ImageOps.exif_transpose(image)

    return image


def github_file(url, save_name = None):
    raw_url = url \
        .replace('://github.com/', '://raw.githubusercontent.com/') \
        .replace('/blob/', '/')
    
    if save_name is None:
        save_name = unquote(os.path.basename(raw_url))
    
    torch.hub.download_url_to_file(raw_url, save_name)
    print('saved file: ' + save_name, file = sys.stderr)

    return save_name


def stock_adobe():
    # request
    url = "https://stock.adobe.com/in/search/images?k=car%20scratch"
    response = requests.get(url)
    # soup
    soup = BeautifulSoup(response.text, "html.parser")
    # images
    images = soup.find_all("img")
    
    data = {}
    for image in images:
        # name
        name = image["alt"]
        if name == "":
            continue
        # link
        try:
            link = image["data-lazy"]
        except:
            link = image["src"]
        if link[-3:] in ["svg", "gif"]:
            continue
        
        if not data:
            data[name] = link
            continue
        
        if name in data.keys():
            name = name + "_1"
        
        data[name] = link
    for name, link in data.items():
        # get images
        image_path = os.path.join(
            "./data/car_scratch/", 
            name.replace(" ", "-").replace("//-", "").replace("-.-", "-").replace(".-", "-").replace("...-", "-").replace("...", "-") + ".jpg"
        )
        if not os.path.exists(image_path):
            with open(image_path, "wb") as f:
                img = requests.get(link)
                f.write(img.content)
                print(f"downloaded: {name, link}")






# 测试代码 main 函数
def main():
    stock_adobe()

if __name__ == "__main__":
    main()
