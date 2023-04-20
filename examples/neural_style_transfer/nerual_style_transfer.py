#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import numpy as np 
import time
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

import tensorflow as tf 
keras = tf.keras

from keras.preprocessing import image
from keras.applications import vgg19
from keras import backend as K


base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data")
models_dir = os.path.join(base_dir, "models")
images_dir = os.path.join(base_dir, "images")



# -------------------------------------------------------
# data 
# -------------------------------------------------------
# 目标图像路径
target_image_path = os.path.join(data_dir, "tubingen.jpg")
# 风格参考图像路径
style_reference_image_path = os.path.join(data_dir, "starry_night_google.jpg")
# 生成图像路径
combination_image_path = os.path.join(images_dir, "combination_image.jpg")


# 生成图像的尺寸
width, height = image.load_img(target_image_path).size
print("target_image width: %s, height: %s." % (width, height))
img_height = 400
img_width = int(width * img_height / height)
print("combination_image width: %s, height: %s." % (width, height))


def preprocessing_image(image_path):
    img = image.load_img(image_path, target_size = (img_height, img_width))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = vgg19.preprocess_input(img)

    return img


def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")

    return x


target_image = K.constant(preprocessing_image(target_image_path))
style_reference_image = K.constant(preprocessing_image(style_reference_image_path))
combination_image = K.placeholder((1, img_height, img_width, 3))

input_tensor = K.concatenate([target_image, style_reference_image, combination_image], axis = 0)
model = vgg19.VGG19(input_tensor = input_tensor, weights = "imagenet", include_top = False)
print("Model loaded.")


# 内容损失
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# 风格损失
def gram_matrix(x):
    """
    Gram Maxtrix

    Arguments:
        x {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))

    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width

    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


# 总变差损失(total variation loss)/正则化损失
def total_variation_loss(x):
    a = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    
    return K.sum(K.pow(a + b, 1.25))




outputs_dict = dict([layer.name, layer.output] for layer in model.layers)
content_layer = "block5_conv2"
style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

loss = K.variable(0.)
layer_features = outputs_dict["content_layer"]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features, combination_features)

for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_image)

grads = K.gradients(loss, combination_image)[0]
fetch_loss_and_grads = K.function([combination_image], [loss, grads])

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_value = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype("float64")
        self.loss_value = loss_value
        self.grad_values = grad_values

        return self.loss_value
    
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None

        return grad_values
    
evaluator = Evaluator()

result_prefix = "my_result"
iterations = 20

x = preprocessing_image(target_image_path)
x = x.flatten()
for i in range(iterations):
    print("Start of iteration", i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loos, x, fprime = evaluator.grades, maxfun = 20)
    print("Current loss value:", min_val)

    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + "_at_iteration_%d.png" % i
    imsave(fname, img)
    print("Image save as", fname)
    end_time = time.time()
    print("Iteration %d completed in %ds" % (i, end_time - start_time))

