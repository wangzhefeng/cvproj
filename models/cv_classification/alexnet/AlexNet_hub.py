# -*- coding: utf-8 -*-


# ***************************************************
# * File        : AlexNet_hub.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-26
# * Version     : 0.1.032600
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
from PIL import Image
from torchvision import transforms


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# data
# ------------------------------
# ImageNet image
# --------------
# image path
imageurl, imagename = (
    "https://github.com/pytorch/hub/raw/master/images/dog.jpg", 
    "./data/dog.jpg"
)
# download image
if not os.path.exists(imagename):
    torch.hub.download_url_to_file(imageurl, imagename)
# load image
input_image = Image.open(imagename)

# ImageNet labels
# ---------------
# labels path
labelurl, labelname, labelname_path = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", 
    "imagenet_classes.txt",
    "./data/imagenet_classes.txt",
)
# labels download
if not os.path.exists(labelname_path):
    os.system(f"wget {labelurl} ; mv {labelname} {labelname_path}")

# labels load
with open(labelname_path, "r") as f:
    categories = [s.strip() for s in f.readlines()]

# data preprocessing
# ------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    )
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)
input_batch.to(device)

# ------------------------------
# model
# ------------------------------
model = torch.hub.load(
    "pytorch/vision:v0.10.0", 
    "alexnet", 
    weights = "AlexNet_Weights.DEFAULT"
)
model.eval()
model.to(device)

# model inference
with torch.no_grad():
    output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim = 0)

# show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
