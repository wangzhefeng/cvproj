# -*- coding: utf-8 -*-


# ***************************************************
# * File        : autoencoder.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-25
# * Version     : 0.1.032520
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]



#===========================================================
#                        codeing
#===========================================================
# 加载数据
(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape)
print(x_test.shape)

# 数据预处理
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# 添加随机白噪声，并限制加噪声后像素仍处于0至1之间
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc = 0.0, scale = 1.0, size = x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)


# 加噪后的图像效果
n = 10
plt.figure(figsize = (20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# 定义模型输入
input_img = Input(shape = (28, 28, 1,))  # N * 28 * 28 * 1


# 28 * 28 * 1
x = Conv2D(32, (3, 3), padding = "same", activation = "relu")(input_img) # 28 * 28 * 32
x = MaxPooling2D((2, 2), padding = "same")(x)                            # 14 * 14 * 32
x = Conv2D(32, (3, 3), padding = "same", activation = "relu")(x)         # 14 * 14 * 32
encoded = MaxPooling2D((2, 2), padding = "same")(x)                      # 7 * 7 * 32

# 7 * 7 * 32
x = Conv2D(32, (3, 3), padding = "same", activation = "relu")(encoded)  # 7 * 7 * 32
x = UpSampling2D((2, 2))(x)                                             # 14 * 14 * 32
x = Conv2D(32, (3, 3), padding = "same", activation = "relu")(x)        # 14 * 14 * 32
x = UpSampling2D((2, 2))(x)                                             # 28 * 28 * 32)
decoded = Conv2D(1, (3, 3), padding = "same", activation = "sigmoid")(x)# 28 * 28 * 1


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer = "adadelta", loss = "binary_crossentropy")


# autoencoder.fit(x_train_noisy, x_train, 
#                 epochs = 100, 
#                 batch_size = 128, 
#                 shuffle = True, 
#                 validation_data = (x_test_noisy, x_test))
# autoencoder.save("autoencoder.h5")

autoencoder = load_model("autoencoder.h5")
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize = (20, 6))
for i in range(n):
    # disploy original 
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display testing dataset
    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
