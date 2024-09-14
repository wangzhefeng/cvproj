# -*- coding: utf-8 -*-

# ***************************************************
# * File        : params.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-14
# * Version     : 1.0.091419
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# hyperparameters
# ------------------------------
batch_size = 100
learning_rate = 0.1
n_iters = 3000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))



import numpy as np
# params
num_epochs = 100
learning_rate = 0.01
# ------------------------------
# data
# ------------------------------
# x
x_values = range(11)
x_train = np.array(x_values, dtype = np.float32)
x_train = x_train.reshape(-1, 1)
# y
y_values = [2 * i + 1 for i in x_values]
y_train = np.array(y_values, dtype = np.float32)
y_train = y_train.reshape(-1, 1)

print(x_train)
print(y_train)
# ------------------------------
# model training
# ------------------------------
# model
model = LinearRegressor(input_dim = 1, output_dim = 1).to(device)

# loss
loss_fn = nn.MSELoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# model training
for epoch in range(num_epochs):
    epoch += 1
    # numpy array -> torch Variable
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)
    # clear gradients w.r.t parameters
    optimizer.zero_grad()
    # forward
    outputs = model(inputs)
    # loss
    loss = loss_fn(outputs, labels)
    # get gradients w.r.t parameters
    loss.backward()
    # update parameters
    optimizer.step()

    print(f"epoch {epoch}, loss {loss.item()}")

# model predict
prediction = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
print(prediction)
# ------------------------------
# model save
# ------------------------------
save_model = True
if save_model:
    # only parameters
    torch.save(model.state_dict(), "./saved_model/linear_regression.pkl")
# ------------------------------
# model load
# ------------------------------
load_model = False
if load_model:
    model.load_state_dict(torch.load("./saved_model/linear_regression.pkl"))

# ------------------------------
# 
# ------------------------------
train_dataset = None
train_loader = None
test_dataset = None
test_loader = None
# ------------------------------
# hyperparameters
# ------------------------------
batch_size = 100
learning_rate = 0.001
n_iters = 3000
num_epochs = int(n_iters / (len(train_dataset) / batch_size))
# ------------------------------
# model training
# ------------------------------
# model
input_dim = 28 * 28
output_dim = 10
model = LogisticRegressor(input_dim, output_dim).to(device)

# loss
loss_fn = nn.CrossEntropyLoss()

# optimier
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# model training
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # load images as Variable
        images = images.view(-1, 28 * 28).requires_grad_().to(device)
        labels = labels
        # clear gradient
        optimizer.zero_grad()
        # forward
        outputs = model(images)
        # loss
        loss = loss_fn(outputs, labels)
        # get gradient
        loss.backward()
        # update parameters
        optimizer.step()

        iter += 1
        if iter % 500 == 0:
            # accuracy
            correct = 0
            total = 0
            # iterate test dataset
            for images, labels in test_loader:
                images = images.view(-1, 28 * 28).to(device)
                # forward
                outputs = model(images)
                # predict
                _, predicted = torch.max(outputs.data, 1)
                # total number of labels
                total += labels.size(0)
                # total correct predictions
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()
            accuracy = 100 * correct.item() / total
            # print loss
            print(f"Iteration: {iter}. Loss: {loss.item()}. Accuracy: {accuracy}")
# ------------------------------
# model save
# ------------------------------
save_model = False
if save_model:
    torch.save(model.state_dict(), "./saved_models/logistic_regression.pkl")
    
    

# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
