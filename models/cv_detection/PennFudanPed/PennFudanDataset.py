# -*- coding: utf-8 -*-


# ***************************************************
# * File        : PeenFudanDataset.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-01-19
# * Version     : 0.1.011916
# * Description : 图像行人检测和分割, Mask R-CNN
# * Link        : model link: https://www.cis.upenn.edu/~jshi/ped_html/
# *               data link: https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
# *                 - 170 images
# *                 - 345 instances of pedestrians
# * Requirement : numpy
# *               torch
# *               PIL
# ***************************************************


# python libraries
import os
import sys

import numpy as np

import torch
from PIL import Image


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class PennFudanDataset(torch.utils.data.Dataset):
    """
    # file format
    PennFudanPed/
        PedMasks/
            FudanPed00001_mask.png
            FudanPed00002_mask.png
            FudanPed00003_mask.png
            FudanPed00004_mask.png
            ...
        PNGImages/
            FudanPed00001.png
            FudanPed00002.png
            FudanPed00003.png
            FudanPed00004.png
    """
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        image: a PIL Image of size (H, W)
        target: a dict containing the following fields
            - boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, 
              ranging from 0 to W and 0 to H
            - labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
            - image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in 
              the dataset, and is used during evaluation
            - area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric, 
              to separate the metric scores between small, medium and large boxes.
            - iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
            - (optionally) masks (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects
            - (optionally) keypoints (FloatTensor[N, K, 3]): For each one of the N objects, 
              it contains the K keypoints in [x, y, visibility] format, defining the object. 
              visibility=0 means that the keypoint is not visible. Note that for data augmentation, 
              the notion of flipping a keypoint is dependent on the data representation, 
              and you should probably adapt references/detection/transforms.py for your new keypoint representation

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        # ------------------------------
        # load images
        # ------------------------------
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        # image_id
        image_id = torch.tensor([idx])
        # ------------------------------
        # load masks
        # ------------------------------
        # mask
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        mask = Image.open(mask_path)
        mask = np.array(mask)  # convert PIL Image into a numpy array
        # object id
        obj_ids = np.unique(mask)  # instances are encoded as different colors(background is 0)
        obj_ids = obj_ids[1:]  # first id is the background, so remove it
        masks = mask == obj_ids[:, None, None]  # split the color-encoded mask into a set of binary masks
        # boxes
        num_objs = len(obj_ids)  # get bounding box coordinates for each mask
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype = torch.float32)  # convert everything into a torch.Tensor
        # labels
        labels = torch.ones((num_objs,), dtype = torch.int64)  # there is only one class
        # masks
        masks = torch.as_tensor(masks, dtype = torch.unit8)
        # area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # is crowd
        iscrowd = torch.zeros((num_objs,), dtype = torch.int64)  # suppose all instances are not crowd
        # ------------------------------
        # target
        # ------------------------------
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()






