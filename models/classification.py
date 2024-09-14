# -*- coding: utf-8 -*-

# ***************************************************
# * File        : classification.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-15
# * Version     : 1.0.091502
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

import matplotlib.pyplot as plt
from torchvision.transforms.functional import InterpolationMode
# Pre-Trained model
from torchvision.models import MobileNet_V3_Small_Weights
from models.mobilenet.MobileNet_v3 import net as mobilenet
from torchvision.models import ResNet101_Weights
from models.resnet.ResNet101 import net as resnet

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# 测试代码 main 函数
def main():
    # ------------------------------
    # data download and load
    # ------------------------------
    from data_provider.data_download import image_download
    # data
    img_url = "https://cdn.shopify.com/s/files/1/0024/9803/5810/products/583309-Product-0-I-637800179303038345.jpg"
    img_name = "583309-Product-0-I-637800179303038345.jpg"
    img_path = "./data/cv_clf_imgs/"
    image, image_tensor = image_download(img_url, img_name, img_path)
    print(image_tensor)
    # ------------------------------
    # data preprocess
    # ------------------------------
    # crop_size = [224]
    # resize_size = [256]
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # interpolation = InterpolationMode.BILINEAR
    # ------------------------------
    # model predict - MobileNet
    # ------------------------------
    # img preprocess
    mobilenet_preprocess_img = MobileNet_V3_Small_Weights.DEFAULT.transforms()
    resnet_preprocess_img = ResNet101_Weights.DEFAULT.transforms()
    img_mobilenet = mobilenet_preprocess_img(image_tensor).unsqueeze(dim = 0)
    img_resnet = resnet_preprocess_img(image_tensor).unsqueeze(dim = 0)
    print(f"Input Image shape: {img_mobilenet.shape}")
    print(f"Input Image shape: {img_resnet.shape}")
    
    # img predict
    pred_mobilenet = mobilenet(img_mobilenet)
    pred_resnet = resnet(img_resnet)
    print(f"Output Image shape: {pred_mobilenet.shape}")
    print(f"Output Image shape: {pred_resnet.shape}")
    
    # target labels
    cates_mobilenet = MobileNet_V3_Small_Weights.DEFAULT.meta["categories"]
    cates_resnet = ResNet101_Weights.DEFAULT.meta["categories"]
    
    preds_mobilenet = [
        cates_mobilenet[idx]
        for idx in pred_mobilenet.argsort()[0].numpy()
    ][::-1][:3]
    preds_resnet = [
        cates_resnet[idx]
        for idx in pred_resnet.argsort()[0].numpy()[::-1][:3]
    ]
    for pred in preds_mobilenet:
        print(pred)
    
    for pred in preds_resnet:
        print(pred)
    # ------------------------------
    # prediction visualization
    # ------------------------------
    # mobilenet
    fig = plt.figure(figsize = (20, 6))
    for i, img in enumerate([image]):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(img)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.text(0, 0, f"{preds_mobilenet[i]}\n")
    plt.show()
    # resnet
    fig = plt.figure(figsize = (20, 6))
    for i, img in enumerate([image]):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(img)
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        ax.text(0, 0, f"{preds_resnet[i]}\n")
    plt.show()
    
if __name__ == "__main__":
    main()
