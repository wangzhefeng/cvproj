# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2025-06-22
# * Version     : 1.0.062218
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import warnings
warnings.filterwarnings("ignore")
import time

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchvision import transforms
from torchvision.models import (
    vit_b_16,
    ViT_B_16_Weights,
)
from watermark import watermark

from data_provider.data_loader import get_dataloaders_cifar10

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger


def train(train_epochs, model, optimizer, train_loader, valid_loader, device):
    """
    model training
    """
    for epoch in range(train_epochs):
        # train accuracy collector
        train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
        # model train mode
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            # model train mode
            model.train()
            # data to device
            features = features.to(device)
            targets = targets.to(device)
            # forward
            logits = model(features)
            loss = F.cross_entropy(logits, targets)
            # backward
            optimizer.zero_grad()
            loss.backward()
            # update model parameters
            optimizer.step()
            # log
            if batch_idx % 300 == 0:
                logger.info(f"Epoch: {epoch+1:04d}/{train_epochs:04d} | Batch [{batch_idx:04d}/{len(train_loader):04d}, | Loss: {loss:.4f}") 
            # model eval mode
            model.eval()
            # collect train accuracy
            with torch.no_grad():
                pred_labels = torch.argmax(logits, dim=1)
                train_acc.update(pred_labels, targets)
        # ------------------------------
        # model valid
        # ------------------------------
        # model eval mode
        model.eval()
        # model valid
        with torch.no_grad():
            val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
            for (features, targets) in valid_loader:
                # data to device
                features = features.to(device)
                targets = targets.to(device)
                # inference
                logits = model(features)
                # valid accuracy
                pred_labels = torch.argmax(logits, dim=1)
                val_acc.update(pred_labels, targets)
            logger.info(f"Epoch: {epoch+1:04d}/{train_epochs:04d} | Train acc: {train_acc.compute() * 100:.2f}% | Val acc: {val_acc.compute()*100:.2f}%")
            train_acc.reset()
            val_acc.reset()




# 测试代码 main 函数
def main():
    logger.info(watermark(packages="torch,lightning", python=True))
    logger.info(f"Torch CUDA available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    L.seed_everything(123)
    
    # 1.Loading the Dataset
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])
    train_loader, valid_loader, test_loader = get_dataloaders_cifar10(
        batch_size=16,
        num_workers=4,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        validation_fraction=0.1,
    )

    # 2.Initializing the Model
    model = vit_b_16(weights=None)
    # replace output layer
    model.heads.head = nn.Linear(in_features=768, out_features=10)
    # model to device
    model.to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    # 3.Finetuning
    start = time.time()
    train(
        train_epochs=10,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
    )
    end = time.time()
    elapsed = end - start
    logger.info(f"Time elapsed: {elapsed/60:.2f}min")
    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f}GB")

    # 4.Evaluation
    with torch.no_grad():
        # model eval mode
        model.eval()
        # test accuracy
        test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
        for (features, targets) in test_loader:
            # data to device
            features = features.to(device)
            targets = targets.to(device)
            # inference
            outputs = model(features)
            # predict
            pred_labels = torch.argmax(outputs, dim=1)
            # accuracy
            test_acc.update(pred_labels, targets)
    logger.info(f"Test Accuracy: {test_acc.compute()*100:.2f}%")

if __name__ == "__main__":
    main()
