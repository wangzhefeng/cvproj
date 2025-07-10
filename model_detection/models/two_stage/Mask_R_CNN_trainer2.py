# -*- coding: utf-8 -*-

# ***************************************************
# * File        : maskrcnn_trainer.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-26
# * Version     : 0.1.042620
# * Description : description
# * Link        : https://mp.weixin.qq.com/s/UBPbPhewk2sBa8td9wy-CA
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from rcnn.Mask_R_CNN import model

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr_scheduler = None
num_epochs = 2



train_df = None
valid_df = None
DIR_TRAIN = None
image_ids = train_df["image_id"].unique()
print(len(image_ids))
valid_ids = image_ids[-10:]
train_ids = image_ids[:-10]


class ScratchDataset(Dataset):
    """
    转换数据集并返回所需的参数
    """

    def __init__(self, dataframe, image_dir, transforms = None) -> None:
        super().__init__()
        self.image_ids = dataframe["image_id"].unique()
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
    
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.dataframe[self.dataframe["image_id"] == image_id]
        # image read
        image = cv2.imread(f"{self.image_dir}/{image_id}.jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        # boxes area
        boxes = records[["x", "y", "w", "h"]].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[: 2] - boxes[: 0])
        area = torch.as_tensor(area, dtype = torch.float32)
        # labels: one class
        labels = torch.ones((records.shape[0],), dtype = torch.int64)
        # suppose all instances are not crowd
        is_crowd = torch.ones((records.shape[0],), dtype = torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = is_crowd
        if self.transforms:
            sample = {
                "image": image,
                "bboxes": target["boxes"],
                "labels": labels,
            }
            sample = self.transforms(**sample)
            image = sample["image"]
            target["boxes"] = torch.tensor(sample["bboxes"])
        return image, target, image_id
    
    def __len__(self) -> int:
        return self.image_ids.shape[0]


def get_train_transform():
    train_transform = A.Compose(
        [
            A.Flip(0.5),
            ToTensorV2(p = 1.0),
        ], 
        bbox_params = {
            "format": "pascal_voc",
            "label_fields": ["labels"],
        },
    )
    return train_transform


def get_valid_transform():
    valid_transform = A.Compose(
        [
            ToTensorV2(p = 1.0),
        ],
        bbox_params = {
            "format": "pascal_voc",
            "label_fields": ["labels"],
        }
    )
    return valid_transform


def collate_fn(batch):
    return tuple(zip(*batch))


# ------------------------------
# data
# ------------------------------
# data set
train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform())

# data split
indices = torch.randperm(len(train_dataset)).tolist()

# data loader
train_data_loader = DataLoader(
    train_dataset,
    batch_size = 16,
    shuffle = False,
    num_workders = 4,
    collate_fn = collate_fn,
)
valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = 8,
    shuffle = False,
    num_workders = 4,
    collate_fn = collate_fn,
)

images, targets, image_ids = next(iter(train_data_loader))

# ------------------------------
# model training 
# ------------------------------
# loss
loss_fn = None
# optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9, weight_decay = 0.0005)


def train():
    loss_hist = Averager()
    itr = 1
    for epoch in range(num_epochs):
        loss_hist.reset()
        for images, targets, image_ids in train_data_loader:
            images = list(image.to(device) for image in images)
            targets = [
                {
                    k: v.to(device)
                    for k, v in t.items()
                } for t in targets
            ]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = loss.item()
            loss_hist.send(loss_value)
            optimizer.zero_grad()
            losses.backward()
            optimizer.setp()

            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")
            itr += 1
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
        print(f"Epoch #{epoch} loss: {loss_hist.value}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
