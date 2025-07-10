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

import numpy as np
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
print(watermark(packages="torch,lightning", python=True))

from exps.exp_basic import Exp_Basic
from utils.plot_losses import plot_losses
from data_provider.data_loader import get_dataloaders_cifar10

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL

from utils.log_util import logger


class Exp(Exp_Basic):

    def __init__(self, args):
        logger.info(f"{40 * '-'}")
        logger.info("Initializing Experiment...")
        logger.info(f"{40 * '-'}")
        super(Exp, self).__init__(args)
    
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
            validation_fraction=self.args.valid_ratio,
            download=True
        )
        
        return train_loader, valid_loader, test_loader

    def _build_model(self):
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
        model.heads.head = nn.Linear(in_features=768, out_features=self.args.num_classes)
        
        # model to device
        # if fabric is None:
        #     model.to(self.device)
        
        return model

    def launch_fabric(self):
        """
        Launch Fabric
        """
        if self.args.use_amp:
            fabric = Fabric(
                accelerator="cuda", 
                devices=self.args.n_device, 
                precision=self.args.precision
            )
            fabric.launch()
        else:
            fabric = None
        
        return fabric

    def _select_criterion(self):
        """
        loss
        """
        criterion = nn.CrossEntropyLoss()

        return criterion

    def _select_optimizer(self):
        """
        optimizer
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.args.learning_rate
        )

        return optimizer

    def _get_model_path(self, setting):
        """
        模型保存路径
        """
        # 模型保存路径
        model_path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(model_path, exist_ok=True)
        # 最优模型保存路径
        model_checkpoint_path = os.path.join(model_path, "checkpoint.pth")
        
        return model_checkpoint_path

    def _get_test_results_path(self, setting):
        """
        结果保存路径
        """
        results_path = os.path.join(self.args.test_results, setting)
        os.makedirs(results_path, exist_ok=True)
        
        return results_path

    def train(self, fabric=None):
        """
        model training
        """
        train_start_time = time.time()
        # model checkpoint path
        model_checkpoint_path = self._get_model_path(setting="")
        # test results path
        test_results_path = self._get_test_results_path(setting="")
        # dataloader
        train_loader, valid_loader, test_loader = self._get_data()
        if fabric is not None:
            train_loader, valid_loader, test_loader = fabric.setup_dataloaders(
                train_loader, valid_loader, test_loader
            )
        # loss
        criterion = self._select_criterion()
        # optimizer
        optimizer = self._select_optimizer()
        # lightning setup
        if fabric is not None:
            self.model, optimizer = fabric.setup(self.model, optimizer)
        # loss
        train_losses, valid_losses = [], []
        # model training
        for epoch in range(self.args.train_epochs):
            # train accuracy collector
            if fabric is not None:
                train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.args.num_classes).to(fabric.device)
            else:
                train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.args.num_classes).to(self.device)
            # train/valid loss
            train_loss, valid_loss = [], []
            # model train mode
            self.model.train()
            for batch_idx, (features, targets) in enumerate(train_loader):
                # model train mode
                self.model.train()
                # data to device
                if fabric is None:
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                # forward
                logits = self.model(features)
                loss = criterion(logits, targets)
                train_loss.append(loss)
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
                    logger.info(f"Epoch: {epoch+1:04d}/{self.args.train_epochs:04d} | Batch [{batch_idx:04d}/{len(train_loader):04d}] | Train Loss: {loss:.4f}")
                # model eval mode
                self.model.eval()
                # collect train accuracy update
                with torch.no_grad():
                    pred_labels = torch.argmax(logits, dim=1)
                    train_acc.update(pred_labels, targets)
            # average train loss
            train_loss = np.average(train_loss.cpu().numpy())
            # model valid
            valid_loss, valid_acc = self.vali(fabric, valid_loader, criterion)
            # log
            logger.info(f"Epoch: {epoch+1:04d}/{self.args.train_epochs:04d} | Train acc: {train_acc.compute() * 100:.2f}% | Val acc: {valid_acc.compute()*100:.2f}%")
            # accuracy reset
            train_acc.reset()
            valid_acc.reset()
        # losses collect
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # log
        train_end_time = time.time()
        elapsed = train_end_time - train_start_time
        if fabric is not None:
            fabric.print(f"Time elapsed: {elapsed/60:.2f}min")
            fabric.print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.2f}GB")
        else:
            logger.info(f"Time elapsed: {elapsed/60:.2f}min")
            logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.2f}GB")
        
        # plot losses
        logger.info("Plot and save train/valid losses...")
        plot_losses(
            train_epochs=self.args.train_epochs,
            train_losses=train_losses, 
            vali_losses=valid_losses, 
            label="loss",
            results_path=test_results_path
        )
        
        # save model
        logger.info("Saving best model...")
        torch.save(self.model.state_dict(), model_checkpoint_path) 
        
        return self.model

    def vali(self, fabric, valid_loader, criterion):
        """
        model valid
        """
        # valid loss
        valid_loss = []
        # model eval mode
        self.model.eval()
        # model inference
        with torch.no_grad():
            # valid accuracy collector
            if fabric is not None:
                valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.args.num_classes).to(fabric.device)
            else:
                valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.args.num_classes).to(self.device) 
            
            for (features, targets) in valid_loader:
                # data to device
                if fabric is None:
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                # inference
                logits = self.model(features)
                loss = criterion(logits, targets)
                valid_loss.append(loss)
                # valid accuracy update
                pred_labels = torch.argmax(logits, dim=1)
                valid_acc.update(pred_labels, targets) 
        # valid loss
        valid_loss = np.average(valid_loss.cpu().numpy())
        # model train mode
        self.model.train()

        return valid_loss, valid_acc

    def test(self, fabric, load=False):
        # dataloader
        train_loader, valid_loader, test_loader = self._get_data()
        if fabric is not None:
            train_loader, valid_loader, test_loader = fabric.setup_dataloaders(
                train_loader, valid_loader, test_loader
            )
        # model checkpoint path
        model_checkpoint_path = self._get_model_path(setting="")
        # load model
        if load:
            logger.info("Loading best model...")
            self.model.load_state_dict(torch.load(model_checkpoint_path))
        # inference
        with torch.no_grad():
            # model eval mode
            self.model.eval()
            # test accuracy
            if fabric is not None:
                test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.args.num_classes).to(fabric.device)
            else:
                test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.args.num_classes).to(self.device)
            
            for (features, targets) in test_loader:
                # data to device
                if fabric is None:
                    features = features.to(self.device)
                    targets = targets.to(self.device)
                # inference
                outputs = self.model(features)
                # predict accuracy
                pred_labels = torch.argmax(outputs, dim=1)
                test_acc.update(pred_labels, targets)
        # log
        if fabric is not None:
            fabric.print(f"Test Accuracy: {test_acc.compute()*100:.2f}%")
        else:
            logger.info(f"Test Accuracy: {test_acc.compute()*100:.2f}%")




# 测试代码 main 函数
def main():
    # params
    args = {
        "train_epochs": 3,
        "batch_size": 16,
        "num_workers": 0,
        "valid_ratio": 0.1,
        "learning_rate": 5e-5,
        "num_classes": 10,
        "pretrain_model": True,
        "compile_model": False,
        "use_amp": False,
        "n_device": 1,
        "precision": "bf16-mixed",
        "checkpoints": "./saved_results/pretrained_models",
        "test_results": "./saved_results/test_results",
        "use_gpu": True,
        "gpu_type": "cuda",
        "use_multi_gpu": 0,
        "devices": "0,1,2,3,4,5,6,7",
    }
    from utils.cv.args_tools import DotDict
    args = DotDict(args)
    
    # precision
    # torch.set_float32_matual_precision("medium")
    
    # random seed 
    L.seed_everything(123)
    
    # 1.exp
    exp = Exp(args)
    
    # 3.Launch Fabric
    fabric = exp.launch_fabric()
    
    # 4.Finetuning
    exp.train(fabric)

    # 5.Evaluation
    exp.test(fabric)

if __name__ == "__main__":
    main()
