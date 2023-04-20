# -*- coding: utf-8 -*-


# ***************************************************
# * File        : semantic.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-29
# * Version     : 0.1.032915
# * Description : semantic image segmentation
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import gc
import os
import sys

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torchvision.transforms.functional import (
    pil_to_tensor,
    to_pil_image,
)
from torchvision.utils import draw_segmentation_masks
from Fast_FCN import FCN_ResNet50_Weights
from Fast_FCN import net


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# data
# ------------------------------
# data download
# ----------------
img_urls, img_path = (
    [
        "https://www.luxurytravelmagazine.com/files/593/2/80152/luxury-travel-instagram_bu.jpg",
        "https://www.akc.org/wp-content/uploads/2020/12/training-behavior.jpg",
        "https://images.squarespace-cdn.com/content/v1/519bd105e4b0c8ea540e7b36/1555002210238-V3YQS9DEYD2QLV6UODKL/The-Benefits-Of-Playing-Outside-For-Children.jpg",
    ],
    "./data/cv_image_segmentation",
)
for img_url in img_urls:
    img_file_name = img_url.split("/")[-1]
    if not os.path.exists(os.path.join(img_path, img_file_name)):
        os.system(f"wget {img_url} ; mv {img_file_name} {img_path}")

# data load
# ----------------
def data_load(img_path, img_name, is_plot = False):
    img = Image.open(os.path.join(img_path, img_name))
    if is_plot:
        plt.figure(figsize = (12, 8))
        plt.imshow(img)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show()
    
    return img


holiday = data_load(img_path, "luxury-travel-instagram_bu.jpg", is_plot = False)
kids_playing = data_load(img_path, "The-Benefits-Of-Playing-Outside-For-Children.jpg", is_plot = False)
dog_kid_playing = data_load(img_path, "training-behavior.jpg", is_plot = False)

# ------------------------------
# data preprocessing
# ------------------------------
def data_preprocessing(img):
    # data tensor
    img_tensor_int = pil_to_tensor(img)
    # add batch dim
    preprocess_img = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms(resize_size = None)
    img_tensor_trans = preprocess_img(img_tensor_int).unsqueeze(dim = 0)
    print(f"Image Tensor shape: {img_tensor_trans.shape}")

    return img_tensor_int, img_tensor_trans

holiday_tensor, holiday_tensor_trans = data_preprocessing(holiday)
kids_playing_tensor, kids_playing_tensor_trans = data_preprocessing(kids_playing)
dog_kid_playing_tensor, dog_kid_playing_tensor_trans = data_preprocessing(dog_kid_playing)

# ------------------------------
# model
# ------------------------------
holiday_pred = net(holiday_tensor_trans)
gc.collect()
print(f"Holiday predicts: {holiday_pred.keys()}")

kids_playing_pred = net(kids_playing_tensor_trans)
gc.collect()
print(f"Kids playing predicts: {kids_playing_pred.keys()}")

dog_kid_playing_pred = net(dog_kid_playing_tensor_trans)
gc.collect()
print(f"Dog kid playing predicts: {dog_kid_playing_pred.keys()}")

# ------------------------------
# reslut visualize
# ------------------------------
def get_result(img_pred, object_idx_name, is_plot):
    # 类别
    cates_mapping = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.meta["categories"]
    class_to_idx = {
        class_: idx for (idx, class_) in enumerate(cates_mapping)
    }
    # prediction
    pred_out = img_pred["out"]
    normalized_mask = pred_out.softmax(dim = 1)[0]
    # object index
    object_idxs = {
        "person": class_to_idx["person"],
        "dog": class_to_idx["dog"],
        "background": class_to_idx["__background__"],
    }
    # object plot
    object_img = to_pil_image(normalized_mask[object_idxs[object_idx_name]])
    if is_plot:
        plt.figure(figsize = (12, 8))
        plt.imshow(object_img)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show()
    
    return normalized_mask, object_idxs


def draw_mask(mask_normalized_coeff, img_tensor_int, normalized_mask, object_idx_name, is_plot):
    masks = normalized_mask > mask_normalized_coeff
    masked_img = draw_segmentation_masks(
        img_tensor_int, 
        masks[object_idx_name],
    )
    masked_img_pil = to_pil_image(masked_img)
    if is_plot:
        plt.figure(figsize = (12, 8))
        plt.imshow(masked_img_pil)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.show()


normalized_mask, object_idxs = get_result(
    img_pred = holiday_pred, 
    object_idx_name = "person", 
    is_plot = False,
)
draw_mask(
    mask_normalized_coeff = 0.7, 
    img_tensor_int = holiday_tensor, 
    normalized_mask = normalized_mask, 
    object_idx_name = object_idxs["person"], 
    is_plot = False,
)
draw_mask(
    mask_normalized_coeff = 0.7, 
    img_tensor_int = holiday_tensor, 
    normalized_mask = normalized_mask, 
    object_idx_name = object_idxs["background"], 
    is_plot = False,
)


# img kids_playing
normalized_mask, object_idxs = get_result(
    img_pred = kids_playing_pred, 
    object_idx_name = "person", 
    is_plot = False,
)
draw_mask(
    mask_normalized_coeff = 0.7, 
    img_tensor_int = kids_playing_tensor, 
    normalized_mask = normalized_mask, 
    object_idx_name = object_idxs["person"], 
    is_plot = False, 
)
draw_mask(
    mask_normalized_coeff = 0.7, 
    img_tensor_int = kids_playing_tensor, 
    normalized_mask = normalized_mask, 
    object_idx_name = object_idxs["background"], 
    is_plot = False,
)


# img dog_kid_palying
normalized_mask_person, object_idxs_person = get_result(
    img_pred = dog_kid_playing_pred, 
    object_idx_name = "person", 
    is_plot = False
)
normalized_mask_dog, object_idxs_dog = get_result(
    img_pred = dog_kid_playing_pred, 
    object_idx_name = "dog", 
    is_plot = False
)
draw_mask(
    mask_normalized_coeff = 0.7, 
    img_tensor_int = dog_kid_playing_tensor, 
    normalized_mask = normalized_mask_person, 
    object_idx_name = object_idxs_person["person"], 
    is_plot = True
)
draw_mask(
    mask_normalized_coeff = 0.7, 
    img_tensor_int = dog_kid_playing_tensor, 
    normalized_mask = normalized_mask_person, 
    object_idx_name = object_idxs_person["background"], 
    is_plot = True
)
draw_mask(
    mask_normalized_coeff = 0.1, 
    img_tensor_int = dog_kid_playing_tensor, 
    normalized_mask = normalized_mask_dog, 
    object_idx_name = object_idxs_dog["dog"], 
    is_plot = True
)
draw_mask(
    mask_normalized_coeff = 0.1, 
    img_tensor_int = dog_kid_playing_tensor, 
    normalized_mask = normalized_mask_dog, 
    object_idx_name = object_idxs_dog["background"], 
    is_plot = True
)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
