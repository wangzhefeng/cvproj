# -*- coding: utf-8 -*-


# ***************************************************
# * File        : data_load.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-19
# * Version     : 0.1.031923
# * Description : 使用 tf.keras.datasets 获得数据集并预处理
# *               实现一个简单的 MNISTLoader 类来读取 MNIST 数据集数据
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import os
import numpy as np
import pickle
import gzip
try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')

import tensorflow as tf
from config.config_loader import settings



def data_loader_local():
    """
    从 http://yann.lecun.com/exdb/mnist/下载数据到本地，并载入数据
    """
    dataset_dir = settings["PATH"]["data_path_base"]
    key_file = settings["PATH"]["data_url_base_key_file"]
    img_size = settings["IMAGE"]["flatten_image_size"]
    save_file = settings["PATH"]["data_path_base"] + "/mnist.pkl"

    def _download(file_name):
        file_path = dataset_dir  + "/" + file_name
        if os.path.exists(file_path):
            return
        print("Downloading " + file_name + " ... ")
        urllib.request.urlretrieve(settings["PATH"]["url_base"] + file_name, file_path)
        print("Done")

    def download_mnist():
        for v in key_file.values():
            _download(v)

    def _load_label(file_name):
        file_path = dataset_dir + "/" + file_name
        print("Converting " + file_name + " to NumPy Array ...")
        with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset = 8)
        print("Done")
        
        return labels

    def _load_img(file_name):
        file_path = dataset_dir + "/" + file_name
        print("Converting " + file_name + " to NumPy Array ...")    
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset = 16)
        data = data.reshape(-1, img_size)
        print("Done")
        return data

    def _convert_numpy():
        dataset = {}
        dataset['train_data'] =  _load_img(key_file['train_data'])
        dataset['train_label'] = _load_label(key_file['train_label'])    
        dataset['test_data'] = _load_img(key_file['test_data'])
        dataset['test_label'] = _load_label(key_file['test_label'])
        return dataset

    def init_mnist():
        download_mnist()
        dataset = _convert_numpy()
        print("Creating pickle file ...")
        with open(save_file, 'wb') as f:
            pickle.dump(dataset, f, -1)
        print("Done!")

    def load_mnist():
        """
        读入MNIST数据集
        """
        if not os.path.exists(save_file):
            init_mnist()
        with open(save_file, 'rb') as f:
            dataset = pickle.load(f)
        return (dataset['train_data'], dataset['train_label']), (dataset['test_data'], dataset['test_label']) 

    return load_mnist()


def data_loader_keras():
    """
    载入 keras 中的 mnist 数据集
    """
    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data()
    return (train_data, train_label), (test_data, test_label)


class MNISTLoader():
    """
    载入 mnist 数据集
    """
    def __init__(self, 
                 normalize = True, 
                 flatten = False, 
                 one_hot_label = False, 
                 data_loader_func = data_loader_keras):
        """
        :param normalize: 将图像的像素值正规化为0.0~1.0, defaults to True
        :type normalize: bool, optional
        :param flatten: 是否将图像展开为一维数组, defaults to False
        :type flatten: bool, optional
        :param one_hot_label: one_hot_label为True的情况下, 标签作为one-hot数组返回, defaults to False
        :type one_hot_label: bool, optional
        :param data_loader_func: 载入函数, defaults to data_loader_keras
        :type data_loader_func: _type_, optional
        
        returns:
            self.train_data
            self.train_label
            self.test_data
            self.test_label
            self.num_train_data
            self.num_test_data
        """
        self.normalize = normalize
        self.flatten = flatten
        self.one_hot_label = one_hot_label
        self.data_loader_func = data_loader_func

        # 载入 minst 数据集
        (self.train_data, self.train_label), (self.test_data, self.test_label) = data_loader_func()
        
        # mnist 中的图像默认为 uint8(0~255的数字)。以下代码将其归一化为 0~1 的浮点数，并在最后增加一维作为颜色通道
        if self.normalize:
            self.train_data = self.train_data.astype(np.float32) / 255.0
            self.test_data = self.test_data.astype(np.float32) / 255.0
        
        img_faltten_size = settings["IMAGE"]["image_size"]["width"] * settings["IMAGE"]["image_size"]["height"]
        if self.data_loader_func is data_loader_local:
            if self.flatten:
                self.train_data = self.train_data
                self.test_data = self.test_data
                self.input_shape = (img_faltten_size,)
            elif not self.flatten and tf.keras.backend.image_data_format() == 'channels_first':
                self.train_data = self.train_data.reshape(-1, 1, 28, 28)
                self.test_data = self.test_data.reshape(-1, 1, 28, 28)
                self.input_shape = (1, settings["IMAGE"]["image_size"]["width"], settings["IMAGE"]["image_size"]["height"])
            elif not self.flatten and tf.keras.backend.image_data_format() == 'channels_last':
                self.train_data = self.train_data.reshape(-1, 28, 28, 1)
                self.test_data = self.test_data.reshape(-1, 28, 28, 1)
                self.input_shape = (settings["IMAGE"]["image_size"]["width"], settings["IMAGE"]["image_size"]["height"], 1)
        elif self.data_loader_func is data_loader_keras:
            if self.flatten:
                self.train_data = self.train_data.reshape(self.train_data.shape[0], img_faltten_size)
                self.test_data = self.test_data.reshape(self.test_data.shape[0], img_faltten_size)
                self.input_shape = (img_faltten_size,)
            elif tf.keras.backend.image_data_format() == 'channels_first':
                # --------------------
                # train data
                # (60000, 1, 28, 28)
                # --------------------
                # method 1
                # self.train_data = self.train_data..reshape(
                #     self.train_data.shape[0], 
                #     1, 
                #     settings["IMAGE"]["image_size"]["width"], 
                #     settings["IMAGE"]["image_size"]["height"],
                # )
                # method 2
                self.train_data = np.expand_dims(self.train_data, axis = 1)
                # method 3
                # self.train_data = self.train_data[..., tf.newaxis]
                # --------------------
                # test data
                # [10000, 1, 28, 28]
                # --------------------
                # self.test_data = self.test_data.reshape(
                #     self.test_data.shape[0], 
                #     1, 
                #     settings["IMAGE"]["image_size"]["width"], 
                #     settings["IMAGE"]["image_size"]["height"],
                # )
                # method 2
                self.test_data = np.expand_dims(self.test_data, axis = 1)
                # method 3
                # self.test_data = self.test_data[..., tf.newaxis]
                # --------------------
                # input_shape
                # --------------------
                self.input_shape = (1, settings["IMAGE"]["image_size"]["width"], settings["IMAGE"]["image_size"]["height"])
            else:
                # --------------------
                # train data
                # [60000, 28, 28, 1]
                # --------------------
                # method 1
                # self.train_data = self.train_data.reshape(
                #     self.train_data.shape[0], 
                #     settings["IMAGE"]["image_size"]["width"], 
                #     settings["IMAGE"]["image_size"]["height"],
                #     1
                # )
                # method 2
                self.train_data = np.expand_dims(self.train_data, axis = 3)
                # method 3
                # self.train_data = self.train_data[..., tf.newaxis]
                # method 4
                # self.train_data = self.train_data.reshape(-1, 1, 28, 28)
                # --------------------
                # test data
                # [10000, 28, 28, 1]
                # --------------------
                # method 1
                # self.test_data = self.test_data.reshape(
                #     self.test_data.shape[0], 
                #     settings["IMAGE"]["image_size"]["width"], 
                #     settings["IMAGE"]["image_size"]["height"], 
                #     1
                # )
                # method 2
                self.test_data = np.expand_dims(self.test_data, axis = 3)
                # method 3
                # self.test_data = self.test_data[..., tf.newaxis]
                # method 4
                # self.test_data = self.test_data.reshape(-1, 1, 28, 28)
                # --------------------
                # input shape
                # --------------------
                self.input_shape = (settings["IMAGE"]["image_size"]["width"], settings["IMAGE"]["image_size"]["height"], 1)
        
        if self.one_hot_label:
            self.train_label = self._change_one_hot_label(self.train_label)
            self.test_label = self._change_one_hot_label(self.test_label)
        else:
            self.train_labels = tf.keras.utils.to_categorical(self.train_label.astype(np.int32), settings["DATA"]["num_classes"]) # (60000,)
            self.test_labels = tf.keras.utils.to_categorical(self.test_label.astype(np.int32), settings["DATA"]["num_classes"]) # (10000,)
        # --------------------
        # trian_data, test_data number
        # --------------------
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        """
        从数据中随机取出 batch_size 个元素并返回
        """
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]

    def data_shuffle(self, shuffle = 10000, shuffle_batch = 32):
        """
        # shuffle datasets

        :param shuffle: _description_, defaults to 10000
        :type shuffle: int, optional
        :param shuffle_batch: _description_, defaults to 32
        :type shuffle_batch: int, optional
        """
        self.train_ds = tf.data.Dataset \
            .from_tensor_slices((self.train_data, self.train_label)) \
            .shuffle(shuffle) \
            .batch(shuffle_batch)
        self.test_ds = tf.data.Dataset \
            .from_tensor_slices((self.test_data, self.test_label)) \
            .batch(shuffle_batch)

    @staticmethod
    def _change_one_hot_label(X):
        T = np.zeros((X.size, 10))
        for idx, row in enumerate(T):
            row[X[idx]] = 1
        return T




data_loader = MNISTLoader(
    normalize = True,
    flatten = False, 
    one_hot_label = False, 
    data_loader_func = data_loader_keras,
)




def main():
    # (train_data, train_label), (test_data, test_label) = data_loader_local()
    # (train_data, train_label), (test_data, test_label) = data_loader_keras()
    # print("train_data.shape:", train_data.shape)
    # print("train_label.shape:", train_label.shape)
    # print("test_data.shape:", test_data.shape)
    # print("test_label.shape:", test_label.shape)

    data_loader = MNISTLoader(
        normalize = True,
        flatten = False, 
        one_hot_label = False, 
        data_loader_func = data_loader_local)
    print("train_data.shape:", data_loader.train_data.shape)
    print("train_label.shape:", data_loader.train_label.shape)
    print("num_train_data:", data_loader.num_train_data)
    print("test_data.shape:", data_loader.test_data.shape)
    print("test_label.shape:", data_loader.test_label.shape)
    print("num_test_data:", data_loader.num_test_data)
    print("input_shape:", data_loader.input_shape)




if __name__ == "__main__":
    main()

