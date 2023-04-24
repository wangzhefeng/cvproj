# -*- coding: utf-8 -*-

# ***************************************************
# * File        : fasterrcnn.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-28
# * Version     : 0.1.032805
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os

from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import torch
from torch import nn
from torchvision.transforms.functional import (
    pil_to_tensor,
    to_pil_image,
)
from torchvision.models.detection import (
    faster_rcnn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fcos,
    fcos_resnet50_fpn,
    FCOS,
    FCOS_ResNet50_FPN_Weights,
    retinanet,
    retinanet_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
    RetinaNet,
    RetinaNet_ResNet50_FPN_Weights,
    RetinaNet_ResNet50_FPN_V2_Weights,
    ssd300_vgg16,
    SSD300_VGG16_Weights,
    ssdlite,
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.utils import draw_bounding_boxes

from Faster_R_CNN import net

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# data
# ------------------------------
# data download
img_urls, img_path= (
    [
        "https://images.click.in/classifieds/images/95/30_12_2017_15_58_25_562aaa7a9b6593ce55f7e59cae781674_vpwodzncbi.jpg",
        "https://gumlet.assettype.com/freepressjournal/import/2016/07/kids-playing.jpg",
    ],
    "./data/object_detection",
)
# for img_url in img_urls:
#     img_name = img_url.split("/")[-1]
#     os.system(f"cd {img_path}; mv 30_12_2017_15_58_25_562aaa7a9b6593ce55f7e59cae781674_vpwodzncbi.jpg holiday.jpg")
#     if not os.path.exists(os.path.join(img_path, img_name)):
#         os.system(f"wget {img_url} ; mv {img_name} {img_path}")

# data load
holiday = Image.open(f"{img_path}/holiday.jpg")
kids_playing = Image.open(f"{img_path}/kids-playing.jpg")

# data view
# fig = plt.figure(figsize = (12, 8))
# plt.imshow(holiday)
# plt.xticks([], [])
# plt.yticks([], [])
# plt.show()

# data tensor
holiday_tensor_int = pil_to_tensor(holiday)
kids_palying_tensor_int = pil_to_tensor(kids_playing)
print(holiday_tensor_int.shape)
print(kids_palying_tensor_int.shape)

# add batch dim
holiday_tensor_int = holiday_tensor_int.unsqueeze(dim = 0)
kids_palying_tensor_int = kids_palying_tensor_int.unsqueeze(dim = 0)
print(holiday_tensor_int.shape)
print(kids_palying_tensor_int.shape)

# data convert
holiday_tensor_float = holiday_tensor_int / 255.0
kids_palying_tensor_float = kids_palying_tensor_int / 255.0
print(holiday_tensor_float.min(), holiday_tensor_float.max())
print(kids_palying_tensor_float.min(), kids_palying_tensor_float.max())

# ------------------------------
# model prediction
# ------------------------------
# predict
holiday_preds = net(holiday_tensor_float)
print(holiday_preds)

# predict filter
holiday_preds[0]["boxes"] = holiday_preds[0]["boxes"][holiday_preds[0]["scores"] > 0.8]
holiday_preds[0]["labels"] = holiday_preds[0]["labels"][holiday_preds[0]["scores"] > 0.8]
holiday_preds[0]["scores"] = holiday_preds[0]["scores"][holiday_preds[0]["scores"] > 0.8]
print(holiday_preds)

# ------------------------------
# result visualize
# ------------------------------
# load target classes mapping
coco_url, coco_path = [
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "./data/COCO",
]
coco_file_name = coco_url.split("/")[-1]
if not os.path.exists(os.path.join(coco_path, coco_file_name)):
    os.system(f"wget {coco_url} ; mv {coco_file_name} {coco_path}")
    os.system(f"cd {coco_path} ; unzip {coco_file_name}")

# coco instance
annFile = "./data/COCO/annotations/instances_val2017.json"
coco = COCO(annFile)

# map target category ids to labels
holiday_labels = coco.loadCats(holiday_preds[0]["labels"].numpy())
print(holiday_labels)

# visualize bounding boxes on images
holiday_annot_labels = [
    "{}-{:.2f}".format(label["name"], prob) 
    for label, prob in zip(holiday_labels, holiday_preds[0]["scores"].detach().numpy())
]
holiday_output = draw_bounding_boxes(
    image = holiday_tensor_int[0],
    boxes = holiday_preds[0]["boxes"],
    labels = holiday_annot_labels,
    colors = ["red" if label["name"] == "person" else "green" for label in holiday_labels],
    width = 2,
)
print(holiday_output.shape)

final_result = to_pil_image(holiday_output)
fig = plt.figure(figsize = (12, 8))
plt.imshow(final_result)
plt.show()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
