# -*- coding: utf-8 -*-


# ***************************************************
# * File        : AlexNet.py
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
from torchvision import transforms
from PIL import Image
import gradio as gr


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# data
# ------------------------------
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

# ------------------------------
# model
# ------------------------------
# model
model = torch.hub.load(
    "pytorch/vision:v0.10.0", 
    "alexnet", 
    weights = "AlexNet_Weights.DEFAULT"
)
model.eval()
model.to(device)

# ------------------------------
# model inference
# ------------------------------
def inference(input_image):
    # ------------------------------
    # data preprocess
    # ------------------------------
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
    # ---------------
    # inference
    # ---------------
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim = 0)
    # ---------------
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
    # ----------------
    # show top categories per image
    # ----------------
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    result = {}
    for i in range(top5_prob.size(0)):
        result[categories[top5_catid[i]]] = top5_prob[i].item()
    
    return result


# ------------------------------
# app
# ------------------------------
inputs = gr.inputs.Image(type = "pil")
outputs = gr.outputs.Label(type = "confidences", num_top_classes = 5)
title = "ALEXNET"
description = "Gradio demo for Alexnet, the 2012 ImageNet winner achieved a top-5 error of 15.3%, more than 10.8 percentage points lower than that of the runner up. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/1404.5997'>One weird trick for parallelizing convolutional neural networks</a> | <a href='https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py'>Github Repo</a></p>"
examples = [
    ["./data/dog.jpg"]
]

gr.Interface(
    fn = inference, 
    inputs = inputs, 
    outputs = outputs, 
    title = title, 
    description = description, 
    article = article, 
    examples = examples, 
    analytics_enabled = False
).launch()






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
