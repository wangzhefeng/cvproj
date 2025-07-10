# -*- coding: utf-8 -*-

# ***************************************************
# * File        : instance.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-29
# * Version     : 0.1.032915
# * Description : instance image segmentation
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
_path = os.path.abspath(os.path.dirname(__file__))
if os.path.join(_path, "..") not in sys.path:
    sys.path.append(os.path.join(_path, ".."))
import gc

from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import (
    pil_to_tensor,
    to_pil_image,
)
from torchvision.utils import draw_segmentation_masks
from models.two_stage.Mask_R_CNN import MaskRCNN_ResNet50_FPN_Weights
from models.two_stage.Mask_R_CNN import net

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# data path
# ------------------------------
img_urls, img_path = (
    [
        "https://www.luxurytravelmagazine.com/files/593/2/80152/luxury-travel-instagram_bu.jpg",
        "https://www.akc.org/wp-content/uploads/2020/12/training-behavior.jpg",
        "https://images.squarespace-cdn.com/content/v1/519bd105e4b0c8ea540e7b36/1555002210238-V3YQS9DEYD2QLV6UODKL/The-Benefits-Of-Playing-Outside-For-Children.jpg",
    ],
    "/Users/zfwang/learn/machinelearning/data/image_seg_det/maskrcnn_resnet50/",
)

# ------------------------------
# data
# ------------------------------
def data_download():
    for img_url in img_urls:
        img_file_name = img_url.split("/")[-1]
        if not os.path.exists(os.path.join(img_path, img_file_name)):
            os.system(f"wget {img_url} ; mv {img_file_name} {img_path}")


def data_load(img_path, img_name, is_plot = False):
    img_file = os.path.join(img_path, img_name)
    img = Image.open(img_file)
    if is_plot:
        plt.figure(figsize = (12, 8))
        plt.imshow(img)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show()
    
    return img


def data_preprocessing(img):
    # data tensor
    img_tensor = pil_to_tensor(img)
    # add batch dim
    preprocess_img = MaskRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
    img_tensor_trans = preprocess_img(img_tensor).unsqueeze(dim = 0)
    print(f"Image Tensor shape: {img_tensor_trans.shape}")

    return img_tensor_trans, img_tensor


# data download
data_download()

# data load
holiday = data_load(img_path, "luxury-travel-instagram_bu.jpg", is_plot = False)
kids_playing = data_load(img_path, "The-Benefits-Of-Playing-Outside-For-Children.jpg", is_plot = False)
dog_kid_playing = data_load(img_path, "training-behavior.jpg", is_plot = False)

# data preprocessing
holiday_tensor_trans, holiday_tensor = data_preprocessing(img = holiday)
kids_playing_tensor_trans, kids_playing_tensor = data_preprocessing(img = kids_playing)
dog_kid_playing_tensor_trans, dog_kid_playing_tensor = data_preprocessing(img = dog_kid_playing)

# ------------------------------
# model
# ------------------------------
holiday_pred = net(holiday_tensor_trans)
gc.collect()
print(f"Holiday predicts: {holiday_pred[0].keys()}")

kids_playing_pred = net(kids_playing_tensor_trans)
gc.collect()
print(f"Kids playing predicts: {kids_playing_pred[0].keys()}")

dog_kid_playing_pred = net(dog_kid_playing_tensor_trans)
gc.collect()
print(f"Dog kid playing: {dog_kid_playing_pred[0].keys()}")

# ------------------------------
# reslut visualize
# ------------------------------
def get_result(img_tensor_int, img_pred, num_object, is_plot = False):
    # all object categories
    cates_mapping = MaskRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"]
    # img predict result
    masks = img_pred[0]["masks"].squeeze()
    labels = img_pred[0]["labels"]
    # detected objects
    detected_objects = [cates_mapping[label] for label in labels[:num_object]]
    print(f"Detected Object: {detected_objects}")
    print(f"Detected Unique Object: {list(set(detected_objects))}")
    # img predict result visualize
    color_mapping = {
        "person": "tomato", 
        "kite": "dodgerblue", 
        "backpack": "yellow", 
        "sports ball": "green", 
        "dog": "orange", 
        "frisbee": "pink", 
        "baseball glove": "grey"
    }
    colors = [color_mapping[cates_mapping[label]] for label in labels[:num_object]]
    output = draw_segmentation_masks(
        img_tensor_int, 
        masks = masks[:num_object].to(torch.bool), 
        colors = colors
    )
    object_img = to_pil_image(output)
    if is_plot:
        plt.figure(figsize = (12, 8))
        plt.imshow(object_img)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show()


# get_result(
#     img_tensor_int = holiday_tensor, 
#     img_pred = holiday_pred, 
#     num_object = 20, 
#     is_plot = True
# )
# get_result(
#     img_tensor_int = kids_playing_tensor, 
#     img_pred = kids_playing_pred, 
#     num_object = 5, 
#     is_plot = True,
# )
get_result(
    img_tensor_int = dog_kid_playing_tensor,
    img_pred = dog_kid_playing_pred,
    num_object = 3,
    is_plot = True,
)


# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
