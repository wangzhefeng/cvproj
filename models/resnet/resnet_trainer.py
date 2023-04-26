# -*- coding: utf-8 -*-


# ***************************************************
# * File        : ImageClassification.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-28
# * Version     : 0.1.032808
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import gc
import os
import sys
import glob

import matplotlib.pyplot as plt
from PIL import Image
from IPython import display
import ipywidgets

import torch
from torch.nn.functional import softmax
from torchvision.transforms.functional import (
    to_pil_image, 
    pil_to_tensor,
    InterpolationMode,
)


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data
# ------------------------------
# data download
img_urls, img_path = (
    [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Giant_Panda_2004-03-2.jpg/1200px-Giant_Panda_2004-03-2.jpg",
        "https://cdn-wordpress-info.futurelearn.com/wp-content/uploads/unique-animals-australia.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/7/7d/Wildlife_at_Maasai_Mara_%28Lion%29.jpg",
        "https://149366112.v2.pressablecdn.com/wp-content/uploads/2016/11/1280px-monachus_schauinslandi.jpg",
        "https://m.media-amazon.com/images/I/51RxQK7kK0L._SY355_.jpg",
        "https://cdn.shopify.com/s/files/1/0024/9803/5810/products/583309-Product-0-I-637800179303038345.jpg"
    ], 
    "./data/cv_clf_imgs/"
)
for img_url in img_urls:
    img_name = img_url.split("/")[-1].replace("%28", "(").replace("%29", ")")
    exists_status = os.path.exists(os.path.join(img_path, img_name))
    if not exists_status:
        os.system(f"wget {img_url} ; mv {img_name} {img_path}")

# imgs
panda = Image.open(f"{img_path}1200px-Giant_Panda_2004-03-2.jpg")
koala = Image.open(f"{img_path}unique-animals-australia.jpg")
lion = Image.open(f"{img_path}Wildlife_at_Maasai_Mara_(Lion).jpg")
sea_lion = Image.open(f"{img_path}1280px-monachus_schauinslandi.jpg")
wall_clock = Image.open(f"{img_path}51RxQK7kK0L._SY355_.jpg")
digital_clock = Image.open(f"{img_path}583309-Product-0-I-637800179303038345.jpg")

# tensor
panda_int = pil_to_tensor(panda)
koala_int = pil_to_tensor(koala)
lion_int = pil_to_tensor(lion)
sea_lion_int = pil_to_tensor(sea_lion)
wall_clock_int = pil_to_tensor(wall_clock)
digital_clock_int = pil_to_tensor(digital_clock)

# ------------------------------
# Pre-Trained model
# ------------------------------
from ResNet101_pretrain import net as resnet
from torchvision.models import ResNet101_Weights

# ------------------------------
# data preprocess
# ------------------------------
crop_size = [224]
resize_size = [256]
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
interpolation = InterpolationMode.BILINEAR

# ------------------------------
# model predict - ResNet
# ------------------------------
# img preprocess
resnet_preprocess_img = ResNet101_Weights.DEFAULT.transforms()
panda_resnet = resnet_preprocess_img(panda_int).unsqueeze(dim = 0)
print(panda_resnet.shape)

# img predict
panda_preds1 = resnet(panda_resnet)
print(panda_preds1.shape)

# target labels
cates_resnet = ResNet101_Weights.DEFAULT.meta["categories"]
preds_resnet = []
preds_resnet.append([cates_resnet[idx] for idx in panda_preds1.argsort()[0].numpy()[::-1][:3]])
for pred in preds_resnet:
    print(pred)

# prediction visualization
fig = plt.figure(figsize = (20, 6))
for i, img in enumerate([panda]):
    ax = fig.add_subplot(2, 3, i + 1)
    ax.imshow(img)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.text(0, 0, f"{preds_resnet[i]}\n")
plt.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
