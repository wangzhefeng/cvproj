[TOC] 

# 学习计划

- TODO

# Keras 内容分解

1. 数据加载、内置数据集
	- Data loading
	- Built-in small datasets

2. 数据预处理
	- Tensorflow.keras.preprocessing

3. 模型构建、模型加载

	- Model API

	- Keras Applications

4. 神经网络层构建

	- Layers API

5. 模型编译

	- Model API
	- Optimizers
	- Losses
	- Metrics

6. 模型超参数调参

	- KerasTuner

7. 模型训练、模型评估、 模型预测

	- Model API

8. 模型保存

	- Model API

9. 模型部署

10. 其他

	- Mixed Precision
	- Utilties
	- Callbacks API

11. 项目

# 1.数据加载、内置数据集

Keras 接受的数据类型:

- Numpy arrays
- Tensorflow Dataset objects
- Python generators

## 1.1 图像(image)数据

图像数据存放格式：

```
main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
```

```python
dataset = keras.preprocessing.image_dataset_from_directory(
    "path/to/main_directory",
    batch_size = 64,
    image_size = (200, 200),
)
# iterate over the batches yielded by the dataset
for data, labels in dataset:
    print(data.shape)  # (64, 200, 200, 3)
    print(data.dtype)  # float32
    print(labels.shape)  # (64,)
    print(labels.dtype)  # int32

```

## 1.2 文本(text)数据

文本存放格式：

```
main_directory/
...class_a/
......a_image_1.txt
......a_image_2.txt
...class_b/
......b_image_1.txt
......b_image_2.txt
```

生产数据集：

```python
dataset = keras.preprocessing.text_dataset_from_directory(
    "path/to/main_directory",
    batch_size = 64,
)
# iterate over the batches yielded by the dataset
for data, labels in dataset:
    print(data.shape)  # (64,)
    print(data.dtype)  # string
    print(labels.shape)  # (64,)
    print(labels.dtype)  # int32

```

## 1.3 结构化(csv, txt, ...)数据



## 1.4 内置数据集

### 图像分类 -- CIFAR10

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

return (x_train, y_train), (x_test, y_test)
```

### 图像分类 -- CIFAR100

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data(label_mode = label_mode)
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

return (x_train, y_train), (x_test, y_test)
```

### 文本分类 -- IMDB Movie review sentiment(IMDB电影评论情绪)

```python
from tensorflow.keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    path = "imdb.npz",
    num_word = None,
    skip_top = 0,
    maxlen = None,
    seed = 113,
    start_char = 1,
    oov_char = 2,
    index_from = 3
)
```



### 文本分类 --  Reuters newswire topics(路透社新闻专题主题分类)

-   数据

	-   11228 新闻专线
		-   each wire is encoded as a sequence of word indexes
	-   46 主题
-   引用

```python
from tensorflow.keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    path = "reuters.npz",
    num_words = None, 
    skip_top = 0,
    maxlen = None,
    test_spilt = 0.2,
    seed = 113,
    start_char = 1,
    oov_char = 1,
    index_from = 3
)

# 用于编码序列的单词索引
# word_index = {"word": index}
word_index = reuters.get_word_index(path = "reuters_word_index.json")
```

### 图像分类 -- MNIST

```python
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.dataset.mnist.load_data(path = "mnist.npz")
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
```

### 图像分类 -- Fashion-MNIST

- 数据

	- Train set:
		- 60000 28x28 grayscale images
		- 10 fashion categories
	- Test set:
		- 10000 28x28 grayscale images
		- 10 fashion categories
-   类别标签

| Label | Description | 中文描述 |
| ----- | ----------- | -------- |
| 0     | T-shirt/top | T恤      |
| 1     | Trouser     | 裤子     |
| 2     | Pullover    |          |
| 3     | Dress       | 裙子     |
| 4     | Coat        | 外衣     |
| 5     | Sandal      |          |
| 6     | Shirt       | 衬衫     |
| 7     | Sneaker     |          |
| 8     | Bag         | 包       |
| 9     | Ankle boot  |          |


- 引用

```python
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

### 结构化数据回归 -- Boston housing price

- 数据

	- 特征个数: 13
	- 目标变量: median values of the houses at a location(in k\$)

- 引用

```python

from tensorflow.keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data(
    path = "~/.keras/datasets", 
    seed, 
    test_split
)
```

# 2.数据预处理





# 3.模型构建、模型加载





# 4.神经网络层构建

## 4.1 神经网络层的类型

- 基本层类
- 激活函数层
- 权重
	- 权重初始化
	- 权重正则化
	- 权重约束

- 核心层
- 卷积层
- 池化层
- 循环层
- 预处理层
- 正则化层
- 重塑层
- 合并层
- 局部链接层