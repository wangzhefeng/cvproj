#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
import tensorflow as tf

#===========================================================
#                        codeing
#===========================================================
def the current_time():
	print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time()))))

CONTENT_IMG = "content.jpg"
STYLE_IMG = "style5.jpg"
OUTPUT_DIR = "neural_style_transfer_tensorflow/"

if not os.path.exists(OUTPUT_DIR):
	od.mkdir(OUTPUT_DIR)

IMAGE_W = 800
IMAGE_H = 600
COLOR_C = 3

NOISE_RATIO = 0.7
BETA = 5
ALPHA = 100
VGG_MODEL = "imagenet_vgg_verydeep-19.mat"
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))



