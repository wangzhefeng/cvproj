# -*- coding: utf-8 -*-
import os
import numpy as np
from keras import layers, models
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt


root_dir = "/Users/zfwang/project/machinelearning/deeplearning"
data_dir = os.path.join(root_dir, "data/cats_vs_dogs/")
data_path = os.path.join(data_dir, "cats_and_dogs_small")
model_path = os.path.join(project_path, "models")


def model_load():
    """
    model
    """
    model_dir = os.path.join(model_path, "cats_and_dogs_small_2.h5")
    model = load_model(model_dir)
    return model


def image_visual():
    """
    image
    """
    cat1700_img_path = os.path.join(data_path, "test/cat/cat.1700.jpg")
    img = image.load_img(cat1700_img_path, target_size = (150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor /= 255.0
    print(img_tensor.shape)
    plt.imshow(img_tensor[0])
    plt.show()


def channel_visual(model):
    # image
    cat1700_img_path = os.path.join(data_path, "test/cat/cat.1700.jpg")
    img = image.load_img(cat1700_img_path, target_size = (150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor /= 255.0
    layers_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = models.Model(inputs = model.input, outputs = layers_outputs)
    activations = activation_model.predict(img_tensor)
    first_layer_activation = activations[0]
    print(first_layer_activation.shape)
    plt.matshow(first_layer_activation[0, :, :, 4], cmap = "viridis")
    plt.matshow(first_layer_activation[0, :, :, 7], cmap = "viridis")
    plt.show()
    return activations


def all_channel_visual(model, activations):
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                display_grid[col * size:(col + 1) * size, row * size:(row + 1) * size] = channel_image
        scale = 1./size
        plt.figure(figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect = "auto", cmap = "viridis")
        plt.show()



if  __name__ == "__main__":
    model = model_load()
    image_visual()
    activations = channel_visual(model)
    all_channel_visual(model, activations)
