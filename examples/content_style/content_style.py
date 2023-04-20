#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''Neural style transfer with Keras.

Run the script with:
```
python neural_style_transfer.py path_to_your_base_image.jpg path_to_your_reference.jpg prefix_for_results
```

e.g.:
```
python neural_style_transfer.py img/tuebingen.jpg img/starry_night.jpg results/my_result
```

Optional parameters:
```
--iter, To specify the number of iterations the style transfer takes place (Default is 10)
--content_weight, The weight given to the content loss (Default is 0.025)
--style_weight, The weight given to the style loss (Default is 1.0)
--tv_weight, The weight given to the total variation loss (Default is 1.0)

```
It is preferable to run this script on GPU, for speed.
Example result: https://twitter.com/fchollet/status/686631033085677568
# Details
Style transfer consists in generating an image
with the same "content" as a base image, but with the
"style" of a different picture (typically artistic).
This is achieved through the optimization of a loss function
that has 3 components: "style loss", "content loss",
and "total variation loss":
	- The total variation loss imposes local spatial continuity between
the pixels of the combination image, giving it visual coherence.
	- The style loss is where the deep learning keeps in --that one is defined
using a deep convolutional neural network. Precisely, it consists in a sum of
L2 distances between the Gram matrices of the representations of
the base image and the style reference image, extracted from
different layers of a convnet (trained on ImageNet). The general idea
is to capture color/texture information at different spatial
scales (fairly large scales --defined by the depth of the layer considered).
 	- The content loss is a L2 distance between the features of the base
image (extracted from a deep layer) and the features of the combination image,
keeping the generated image close enough to the original one.

# References
    - [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
'''

"""
一个训练好的神经网络
一张风格图像，用来计算它的风格representation，算完就可以扔了
一张内容图像，用来计算它的内容representation，算完扔
一张噪声图像，用来迭代优化
一个loss函数，用来获得loss
一个求反传梯度的计算图，用来依据loss获得梯度修改图片
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from __future__ import print_function
from keras.preprocessing.image import load_img, img_to_array
from scipy.misc import imsave
from scipy.optimizer import fmin_l_bfgs_b
import time
import argparse

from keras.application import vgg16, vgg19
from keras import backend as K


#===========================================================
#                        codeing
#===========================================================


# ***********************************************************
# 				        initation
# ***********************************************************
parse = argparse.ArgumentParser(description = "Neural style transfer with Keras.")
parse.add_argument("base_image_path", 
				   metavar = "base", type = str, 
				   help = "Path to the iamge to transform.")
parse.add_argument("style_reference_image_path", 
				   metavar = "ref", type = str, 
				   help = "Path to the style reference image.")
parse.add_argument("result_prefix", 
				   metavar = "res_prefix", type = str, 
				   help = "Prefix for the saved results.")
parse.add_argument("--iter", type = int, default = 10, required = False, 
				   help = "Number of iterations to run.")
parse.add_argument("--content_weight", type = float, default = 0.025, required = False, 
				   help = "Content weight.")
parse.add_argument("--style_weight", type = float, default = 1.0, required = False, 
				   help = "Style weight.")
parse.add_argument("--tv_weight", type = float, default = 1.0, required = False, 
				   help = "Total Variation weight.")
args = parse.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix
iterations = args.iter
content_weight = args.content_weight
style_weight = args.style_weight
total_variation_weight = args.tv_weight

# Dimensions of generated picture
width, height = load_img(base_image_path).size
image_nrows = 400
image_ncols = int(width * image_nrows / height)


# ***********************************************************
# 				   载入keras中的网络模型
# ***********************************************************
# 载入keras中的VGG16网络模型
model = vgg16.VGG16(input_tensor = input_tensor, weights = "imagenet", include_top = False)
print("Model VGG16 loaded.")

# 载入keras中的VGG19网络模型
moldel = vgg19.VGG19(input_tensro = input_tensor, weights = "imagenet", include_top = False)
print("Model VGG19 loaded.")


# ***********************************************************
# 							数据预处理
# ***********************************************************
# 输入图像预处理及后处理
def preprocess_image(image_path):
	img = load_img(image_path, target_size = (image_nrows, image_ncols))
	img = img_to_array(img)
	img = np.expand_dims(img, axis = 0)
	img = vgg16.preprocess_input(img)
	return img

def deprocess_image(x):
	# if K.image_data_format() == "channels_first":
	if K.image_dim_ordering() == "th":
		x = x.reshape((3, image_nrows, image_ncols))
		x = x.transpose((1, 2, 0))
	else:
		x = reshape((image_nrows, image_ncols, 3))
	# Remove zero-center by mean pixel
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.680
	# 'BGR' => 'RGB'
	x = x[:, :, ::-1]
	x = np.clip(x, 0, 255).astype("uint8")
	return x

def image_input(base_image_path, style_reference_image_path, image_nrows, image_ncols):
	"""
	将content_image, style_image, init_image组成一个tensor作为网络的输入
	"""
	# 读入内容和风格图，包装为keras张量，这是一个常数的四阶张量
	content_image = K.variable(preprocess_image(base_image_path))
	style_reference_image = K.variable(preprocess_image(style_reference_image_path))
	# 初始化一个待优化图片的占位符，这个地方实际跑起来的时候要填一张噪声图片
	if K.image_dim_ordering() == "th":
		combination_image = K.placeholder((1, 3, image_nrows, image_ncols))
	else:
		combination_image = K.placeholder((1, image_nrows, image_ncols, 3))
	# 将三个张量串联到一起，形成一个形如(3, 3, image_nrows, image_ncols)的张量
	input_tensor = K.concatenate([content_image, style_reference_image, combination_image], axis = 0)
	return input_tensor


# ***********************************************************
# 						损失函数定义
# ***********************************************************
# Gramma矩阵计算图
def gram_matrix(x):
	"""
	输入为某一层的representation
	"""
	assert K.ndim(x) == 3
	if K.image_dim_ordering() == "th":
		features = K.batch_flatten(x)
	else:
		features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	gram = K.dot(features, K.transpose(features))
	return gram

# 计算style loss
def style_loss(style, combination):
	"""
	以style image和combination image的representation为输入计算Gramma矩阵
	计算两个Gramma矩阵的差的二范数
	除以一个归一化值
	"""
	assert K.ndim(style) == 3
	assert K.ndim(combination) == 3
	S = gram_matrix(style)
	C = gram_matrix(combination)
	channels = 3
	size = image_nrows * image_ncols
	loss = K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))
	return loss

# 计算content loss
def content_loss(content, combination):
	"""
	以content image和combination image的representation为输入，计算他们的差的二范数
	"""
	loss = K.sum(K.square(combination - content))
	return loss

# 施加全变差正则，全变差正则用于使生成的图片更加平滑自然
def total_variation_loss(x):
	assert K.ndim(x) == 4
	if K.image_dim_ordering() = "th":
		a = K.square(x[:, :, :image_nrows - 1, :image_ncols - 1] - x[:, :, 1:, :image_ncols - 1])
		b = K.square(x[:, :, :image_nrows - 1, :image_ncols - 1] - x[:, :, :image_nrows - 1, 1:])
	else:
		a = K.square(x[:, :image_nrows - 1, :image_ncols - 1, :] - x[:, 1:, :image_ncols - 1, :])
		b = K.square(x[:, :image_nrows - 1, :image_ncols - 1, :] - x[:, :image_nrows - 1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))


# ***********************************************************
# 							获取反向梯度
# ***********************************************************
# 生成一个tensor字典，建立了层名称到层输出张量的映射
outputs_dict = dict([layer.name, layer.output] for layer in model.layer) 
# layer_output = model.get_layer(layer_name).output

# 初始化一个标量(浮点数)张量保存loss的值
loss = K.variable(0.)

# 计算content image representation和combination image representation的content loss
layer_features = outputs_dict["block4_conv2"]
# layer_features = outputs_dict["block5_conv2"]
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features, combination_features)

# 计算各个层的style image representation和combination image representation的style loss
feature_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block_conv1", "block5_conv1"]
for layer_name in feature_layers:
	layer_features = outputs_dict[layer_name]
	style_reference_features = layer_features[1, :, :, :]
	combination_features = layer_features[2, :, :, :]
	sl = style_loss(style_reference_features, combination_features)
	loss += (style_weight / len(features_layers)) * sl

# 计算全变差约束，加入总loss中
loss += total_variation_weight * total_variation_loss(combination_image)

# 通过K.grad获取反传梯度
grads = K.gradients(loss, combination_image)

outputs = [loss]
if type(grads) in {list, tuple}:
# if isinstance(grads, (list, tuple)):
	outputs += grads
else:
	outputs.append(grads)
f_outputs = K.function([combination_image], outputs)


def eval_loss_and_grad(x):
	# 把输入reshape成矩阵
	if K.image_dim_ordering() == "th":
		x = x.reshape(1, 3, image_nrows, image_ncols)
	else:
		x = x.reshape(1, image_nrows, image_ncols, 3)
	outs = f_outputs([x])
	loss_value = outs[0]
	if len(outs[1:]) == 1:
		grad_value = outs[1].flatten().astype("float64")
	else:
		grad_value = np.array(outs[1:]).flatten().astype("float64")
	return loss_value, grad_value


class Evaluator(object):
	def __init__(self):
		self.loss_value = None
		self.grad_values = None

	def loss(self, x):
		assert self.loss_value is None
		loss_value, grad_values = eval_loss_and_grad(x)
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is None
		grad_value = np.copy(self.grad_values)
		self.loss_value = loss_value
		self.grad_values = None
		return grad_values

evaluator = Evaluator()


# ***********************************************************
# 						运行模型
# ***********************************************************
# 根据后端初始化一张噪声图片，做去均值
# if K.image_dim_ordering() == "th":
# 	x = np.random.uniform(0, 255, (1, 3, image_nrows, image_ncols)) - 128
# else:
# 	x = np.random.uniform(0, 255, (1, image_nrows, image_ncols, 3)) - 128


# Run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
for i in range(iterations):
	print("Start of iteration:", i)
	start_time = time.time()

	x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime = evaluator.grads, maxfun = 20)
	print("Current loss value:", min_val)
	
	# save current generated image
	img = deprocess_image(x.copy())
	fname = result_prefix + "_at_iteration_%d.png" % i
	imsave(fname, img)
	
	end_time = time.time()
	print("Image saved as", fname)
	print("Iteration %d completed in %ds" % (i, end_time - start_time))
