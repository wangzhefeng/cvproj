# -*- coding: utf-8 -*-

# ***************************************************
# * File        : vit_pretrained.py
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
from lightning import Fabric
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

from exps.exp_basic import Exp_Basic
from data_provider.data_loader import get_dataloaders_cifar10

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger


class Model(Exp_Basic):

    def __init__(self, args):
        self.args = args
        # exp config
        logger.info(watermark(packages="torch,lightning", python=True))
        logger.info(f"Torch CUDA available: {torch.cuda.is_available()}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _get_data(self):
        """
        get dataloader
        """
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
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            validation_fraction=self.valid_ratio,
            download=True
        )
        
        return train_loader, valid_loader, test_loader

    def _build_model(self, fabric=None):
        """
        Initializing the Model
        """
        # model
        if not self.args.pretrain_model:
            model = vit_b_16(weights=None)
        else:
            model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # model compile
        if self.args.compile_model:
            model = torch.compile(model)
        # replace output layer
        model.heads.head = nn.Linear(in_features=768, out_features=10)
        # model to device
        if fabric is None:
            model.to(self.device)
        
        return model

    def select_optimizer(self):
        """
        optimizer
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        return optimizer

    def train(self, fabric=None):
        """
        model training
        """
        # dataloader
        train_loader, valid_loader, test_loader = self._get_data()
        # optimizer
        optimizer = self.select_optimizer()
        # model training
        for epoch in range(self.args.train_epochs):
            # train accuracy collector
            if fabric is not None:
                train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(fabric.device)
            else:
                train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(self.device)
            # model train mode
            self.model.train()
            for batch_idx, (features, targets) in enumerate(train_loader):
                # model train mode
                self.model.train()
                # data to device
                features = features.to(self.device)
                targets = targets.to(self.device)
                # forward
                logits = self.model(features)
                loss = F.cross_entropy(logits, targets)
                # backward
                optimizer.zero_grad()
                if fabric is not None:
                    fabric.backward(loss)
                else:
                    loss.backward()
                # update model parameters
                optimizer.step()
                # log
                if batch_idx % 300 == 0:
                    logger.info(f"Epoch: {epoch+1:04d}/{self.args.train_epochs:04d} | Batch [{batch_idx:04d}/{len(train_loader):04d}, | Loss: {loss:.4f}") 
                # model eval mode
                self.model.eval()
                # collect train accuracy
                with torch.no_grad():
                    pred_labels = torch.argmax(logits, dim=1)
                    train_acc.update(pred_labels, targets)
            # ------------------------------
            # model valid
            # ------------------------------
            # model eval mode
            self.model.eval()
            # model valid
            with torch.no_grad():
                if fabric is not None:
                    val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(fabric.device)
                else:
                    val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(self.device) 
                for (features, targets) in valid_loader:
                    # data to device
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                    # inference
                    logits = self.model(features)
                    # valid accuracy
                    pred_labels = torch.argmax(logits, dim=1)
                    val_acc.update(pred_labels, targets)
                logger.info(f"Epoch: {epoch+1:04d}/{self.train_epochs:04d} | Train acc: {train_acc.compute() * 100:.2f}% | Val acc: {val_acc.compute()*100:.2f}%")
                train_acc.reset()
                val_acc.reset()

    def test(self, fabric):
        # dataloader
        train_loader, valid_loader, test_loader = self._get_data()
        # inference
        with torch.no_grad():
            # model eval mode
            self.model.eval()
            # test accuracy
            if fabric is not None:
                test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(fabric.device)
            else:
                test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(self.device)
            for (features, targets) in test_loader:
                # data to device
                features = features.to(self.device)
                targets = targets.to(self.device)
                # inference
                outputs = self.model(features)
                # predict
                pred_labels = torch.argmax(outputs, dim=1)
                # accuracy
                test_acc.update(pred_labels, targets)
        if fabric is not None:
            fabric.print(f"Test Accuracy: {test_acc.compute()*100:.2f}%")
        else:
            logger.info(f"Test Accuracy: {test_acc.compute()*100:.2f}%")




# 测试代码 main 函数
def main():
    # params
    pretrain_model = True
    compile_model = False
    use_amp = False
    
    
    L.seed_everything(123)
    
    # 1.Loading the Dataset
    
    
    
    # 2.Initializing the Model
    # optimizer
    
    
    # 3.Launch Fabric
    if use_amp:
        fabric = Fabric(accelerator="cuda", devices=1)
        fabric.launch()
        train_loader, valid_loader, test_loader = fabric.setup_dataloaders(train_loader, valid_loader, test_loader)
        model, optimizer = fabric.setup(model, optimizer)
    else:
        fabric = None
    
    # 4.Finetuning
    start = time.time()
    train(
        train_epochs=3,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        fabric=fabric,
    )
    end = time.time()
    elapsed = end - start 
    if fabric is not None:
        fabric.print(f"Time elapsed: {elapsed/60:.2f}min")
        fabric.print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f}GB")
    else:
        logger.info(f"Time elapsed: {elapsed/60:.2f}min")
        logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f}GB")
    
    # 5.Evaluation
    # TODO

if __name__ == "__main__":
    main()
