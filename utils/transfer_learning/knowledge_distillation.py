# -*- coding: utf-8 -*-

# ***************************************************
# * File        : knowledge_dis.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091415
# * Description : description
# * Link        : https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************


"""
Prerequisties:
- 1 GPU, 4GB of memory
- PyTorch v2.0 or later
- CIFAR-10 dataset(`/data`)
"""

__all__ = []

# python libraries
import os
import sys

import torch.utils
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from data_provider.CIFAR10 import transforms_cifar, get_dataloader

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]
# check if GPU is available, and if not, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device use: {device}.")
# params
batch_size = 128


# ------------------------------
# data
# ------------------------------
train_loader, test_loader = get_dataloader(
    batch_size = batch_size,
    train_transforms = transforms_cifar,
    test_transforms = transforms_cifar,
    num_workers = 1,
)

# ------------------------------
# model
# ------------------------------
class DeepNN(nn.Module):
    """
    Deeper neural network class to be used as teacher.
    """

    def __init__(self, num_classes = 10) -> None:
        super(DeepNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class LightNN(nn.Module):
    """
    Lightweight neural network class to be used as student
    """

    def __init__(self, num_classes = 10) -> None:
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


# ------------------------------
# 
# ------------------------------
def train(model, train_loader, epochs, learning_rate, device):
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    # compute graph
    model.train()
    # model train
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}.")


def test(model, test_loader, device):
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # model inference
            outputs= model(input)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}.")

    return accuracy


def train_knowledge_distillation(teacher, student, train_loader, 
                                 epochs, learning_rate, T, 
                                 soft_target_loss_weight, ce_loss_weight, device):
    """
    知识蒸馏

    Args:
        teacher (_type_): _description_
        student (_type_): _description_
        train_loader (_type_): _description_
        epochs (_type_): _description_
        learning_rate (_type_): _description_
        T (_type_): _description_
        soft_target_loss_weight (_type_): _description_
        ce_loss_weight (_type_): _description_
        device (_type_): _description_
    """
    





# 测试代码 main 函数
def main():
    '''
    # params
    epochs = 10
    learning_rate = 0.001
    
    # ------------------------------
    # 
    # ------------------------------
    # training deepweight network
    torch.manual_seed(42)
    nn_deep = DeepNN(num_classes=10).to(device)
    train(nn_deep, train_loader, epochs, learning_rate, device)
    test_accuracy_deep = test(nn_deep, test_loader, device)

    # instantiate lightweight network
    torch.manual_seed(42)
    nn_light = LightNN(num_classes=10).to(device)

    # instantiate one more lightweight network
    torch.manual_seed(42)
    new_nn_light = LightNN(num_classes=10).to(device)
    
    # inspect the norm of last lightweight network's first layer
    print(f"Norm of 1st layer of nn_light: {torch.norm(nn_light.features[0].weight).item()}")
    print(f"Norm of 1st layer of new_nn_light: {torch.norm(new_nn_light.features[0].weight).item()}")

    # print the total number of parameters in each model
    total_params_deep = "{:,}".format(sum(p.numel() for p in nn_deep.parameters()))
    print(f"DeepNN parameters: {total_params_deep}")
    total_params_light = "{:,}".format(sum(p.numel() for p in nn_light.parameters()))
    print(f"LightNN parameters: {total_params_light}")
    
    # train and test the lightweight network
    train(nn_light, train_loader, epochs, learning_rate, device)
    test_accuracy_light_ce = test(nn_light, test_loader, device)
    print(f"Teacher accuracy: {test_accuracy_deep:.2f}")
    print(f"Student accuracy: {test_accuracy_light_ce:.2f}")
    # ------------------------------
    # knowledge distillation run
    # ------------------------------
    '''
    pass

if __name__ == "__main__":
    main()
