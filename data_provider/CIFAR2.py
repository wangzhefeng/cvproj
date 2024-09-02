# -*- coding: utf-8 -*-

# ***************************************************
# * File        : CIFAR2.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-21
# * Version     : 0.1.042116
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


train_dir = "./cifar2/train/"
test_dir = "./cifar2/test/"


class Cifar2Dataset(Dataset):

    def __init__(self, imgs_dir, img_transform):
        self.files = list(Path(imgs_dir).rglob("*.jpg"))
        self.transform = img_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        # features
        file_i = str(self.files[i])
        img = Image.open(file_i)
        tensor = self.transform(img)
        # labels
        label = torch.tensor([1.0]) if "1_automobile" in file_i else torch.tensor([0.0])

        return tensor, label


# transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomRotation(45),  # 随机在 45 角度内旋转
    transforms.ToTensor(),  # 转换成张量
])
transform_valid = transforms.Compose([
    transforms.ToTensor()
])

# dataset
train_dataset = Cifar2Dataset(
    train_dir, 
    transform_train
)
valid_dataset = Cifar2Dataset(
    test_dir, 
    transform_valid
)

# dataloader
train_dataloader = DataLoader(
    train_dataset, 
    batch_size = 50, 
    shuffle = True,
)
valid_dataloader = DataLoader(
    valid_dataset, 
    batch_size = 50, 
    shuffle = True,
)

# test
for features, labels in train_dataloader:
    print(features.shape)
    print(labels.shape)
    break





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
