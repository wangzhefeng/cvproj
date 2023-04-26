# -*- coding: utf-8 -*-

# ***************************************************
# * File        : torch_save.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-25
# * Version     : 0.1.042522
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
import torch
import torch.onnx as onnx
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# model weight save & load
# ------------------------------
# save
model = models.vgg16(pretrained = True)
torch.save(model.state_dict(), "./model/vgg16_model_weight.pth")
# load
model = models.vgg16()
model.load_state_dict(torch.load("./model/vgg16_model_weight.pth"))
model.eval()

# ------------------------------
# model save & load
# ------------------------------
# save
torch.save(model, "./model/model_vgg16.pth")
# load
model = torch.load("./model/model_vgg16.pth")

# ------------------------------
# onnx
# ------------------------------
input_image = torch.zeros(1, 3, 224, 224)
onnx.export(model, input_image, "./model/model.onnx")


# ------------------------------
# 
# ------------------------------
class NetA(nn.Module):
    
    def __init__(self):
        super(NetA, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class NetB(nn.Module):
    
    def __init__(self):
        super(NetB, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
# model
# ------------------------
net = NetA()
print(f"Model:\n==========\n {net}")
# optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
print(f"Optimizer:\n==========\n {optimizer}")
# print model's state_dict
print("Model's state_dict:\n======================")
for param_tensor in net.state_dict():
    print(f"{param_tensor}, \t, {net.state_dict()[param_tensor].size()}")
# print optimizer's state_dict
print("Optimizer's state_dict:\n=======================")
for var_name in optimizer.state_dict():
    print(f"{var_name} \t, {optimizer.state_dict()[var_name]}")

# model weight save & load
# ------------------------
PATH = "./models/state_dict_model.pt"
# model save
torch.save(net.state_dict(), PATH)
# model load
model = NetA()
model.load_state_dict(torch.load(PATH)) # 必须在将保存的 state_dict 传递给 load_state_dict 之前反序列化它
model.eval()

# entire model save & load
# ------------------------
PATH = "./models/entire_model.pt"
torch.save(net, PATH)
# model load
model = torch.load(PATH)
model.eval()

# checkpoint save & load
# ------------------------
PATH = "./models/checkpoint_model.pt"
EPOCH = 5
LOSS = 0.4
# model save
torch.save(
    {
        'epoch': EPOCH,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': LOSS,
    }, 
    PATH
)
# model load
checkpoint = torch.load(PATH)

model = NetA()
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

epoch = checkpoint["epoch"]

loss = checkpoint["loss"]

model.eval()
# - or - 
model.train()


# multi-model save & load
# ------------------------
# models
netA = NetA()
netB = NetA()

# optimizers
optimizerA = optim.SGD(netA.parameters(), lr = 0.001, momentum = 0.9)
optimizerB = optim.SGD(netB.parameters(), lr = 0.001, momentum = 0.9)

# path of model save
PATH = "./models/model.pt"

# model save
torch.save(
    {
        "modelA_state_dict": netA.state_dict(),
        "modelB_state_dict": netB.state_dict(),
        "optimizerA_state_dict": optimizerA.state_dict(),
        "optimizerB_state_dict": optimizerB.state_dict(),
    }, 
    PATH
)

# model load
modelA = NetA()
modelB = NetA()
optimModelA = optim.SGD(modelA.parameters(), lr = 0.001, momentum = 0.9)
optimModelB = optim.SGD(modelB.parameters(), lr = 0.001, momentum = 0.9)

checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint["modelA_state_dict"])
modelB.load_state_dict(checkpoint["modelB_state_dict"])
optimizerA.load_state_dict(checkpoint["optimizerA_state_dict"])
optimizerB.load_state_dict(checkpoint["optimizerB_state_dict"])

modelA.eval()
modelB.eval()
# - or -
modelA.train()
modelB.train()

# 使用不同模型参数的热启动模型
# --------------------------
PATH = "./models/A_model.pt"
# save model A
torch.save(netA.state_dict(), PATH)
# load model B
netB.load_state_dict(torch.load(PATH), strict = False)


# 跨设备保存、加载模型
# ------------------------
# 在 GPU 上保存、在 CPU 上加载
PATH = "./models/model.pt"
# model save
torch.save(net.state_dict(), PATH)
# model load
device = torch.device("cpu")
model = NetA()
model.load_state_dict(torch.load(PATH, map_location = device))


# 在 GPU 上保存、在 GPU 上加载
PATH = "./models/model.pt"
# model save
torch.save(net.state_dict(), PATH)
# model load
device = torch.device("cuda")
model = NetA()
model.load_state_dict(torch.load(PATH))
model.to(device)


# 在 CPU 上保存、在 GPU 上加载
PATH = "./models/model.pt"
# model save
torch.save(net.state_dict(), PATH)
# model load
device = torch.device("cuda")
model = NetA()
model.load_state_dict(torch.load(PATH, map_location = "cuda:0"))
model.to(device)


# 保存 torch.nn.DataParallel 模型
PATH = "./models/model.pt"
# model save
torch.save(net.module.state_dict(), PATH)
# load to any divce
torch.load(PATH)
# - or -
model = NetA()
model.load_state_dict(torch.load())




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
