# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
# import keras
# from keras import layers, models, optimizers
# from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing import image
# from keras.applications import VGG16
import matplotlib.pyplot as plt

print(f"Using Keras: {keras.__version__}")
print(f"Using TensorFlow: {tensorflow.__version__}")


root_dir = "/Users/zfwang/project/machinelearning/deeplearning"
data_path = os.path.join(root_dir, "data/cats_vs_dogs/")
original_dataset_dir = os.path.join(data_path, "kaggle_original_data")
small_data_dir = os.path.join(data_path, "cats_and_dogs_small")
train_dir = os.path.join(small_data_dir, "train")
validation_dir = os.path.join(small_data_dir, "validation")
test_dir = os.path.join(small_data_dir, "test")
train_cats_dir = os.path.join(small_data_dir, "train/cat")
project_path = os.path.join(root_dir, "deeplearning/src/project_src/cats_vs_dogs")
models_path = os.path.join(project_path, "models")



def mkdir_data_class(base_dir, data_class):
    for s in data_class:
        dir_var = os.path.join(base_dir, s)
        os.mkdir(dir_var)


def mkdir_animal_class(base_dir, data_class, animal_class):
    for s in data_class:
        for a in animal_class:
            second_dir = os.path.join(base_dir, s)
            print(second_dir)
            dir_var = os.path.join(second_dir, a)
            print(dir_var)
            os.mkdir(dir_var)


def split_data(original_dataset_dir, base_dir, animal_class):
    for a in animal_class:
        fnames1 = [a + '.{}.jpg'.format(i) for i in range(1000)]
        fnames2 = [a + '.{}.jpg'.format(i) for i in range(1000, 1500)]
        fnames3 = [a + '.{}.jpg'.format(i) for i in range(1500, 2000)]
        for fname in fnames1:
            src = os.path.join(original_dataset_dir, fname)
            second_dir = os.path.join(base_dir, "train")
            third_dir = os.path.join(second_dir, a)
            dst = os.path.join(third_dir, fname)
            shutil.copyfile(src, dst)
        for fname in fnames2:
            src = os.path.join(original_dataset_dir, fname)
            second_dir = os.path.join(base_dir, "validation")
            third_dir = os.path.join(second_dir, a)
            dst = os.path.join(third_dir, fname)
            shutil.copyfile(src, dst)
        for fname in fnames3:
            src = os.path.join(original_dataset_dir, fname)
            second_dir = os.path.join(base_dir, "test")
            third_dir = os.path.join(second_dir, a)
            dst = os.path.join(third_dir, fname)
            shutil.copyfile(src, dst)


def PrintData(base_dir, data_class, animal_class):
    for s in data_class:
        for a in animal_class:
            second_dir = os.path.join(base_dir, s)
            data_dir = os.path.join(second_dir, a)
            print("Total" + s + a + "images", len(os.listdir(data_dir)))


def data_processinger(model, is_data_augmentation = 0):
    if is_data_augmentation == 0:
        train_datagen = ImageDataGenerator(rescale = 1./255)
    else:
        train_datagen = ImageDataGenerator(
            rescale = 1./255,
            rotation_range = 40,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            fill_mode = "nearest",
        )
    test_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (150, 150),
        batch_size = 20,
        class_mode = "binary"
    )
    for data_batch, labels_batch in train_generator:
        print("training data batch shape:", data_batch.shape)
        print("training labels batch shape:", labels_batch.shape)
        break

    validation_generator = test_datagen.flow_from_directory(
        validation_dir, 
        target_size = (150, 150),
        batch_size = 20,
        class_mode = "binary"
    )
    for data_batch, labels_batch in validation_generator:
        print("training data batch shape:", data_batch.shape)
        print("training labels batch shape:", labels_batch.shape)
        break

    return train_generator, validation_generator


def extract_features(conv_base, directory, sample_count, batch_size = 20):
    features = np.zeros(shape = (sample_count, 4, 4, 512))
    labels = np.zeros(shape = (sample_count))
    datagen = ImageDataGenerator(rescale = 1./255)
    generator = datagen.flow_from_directory(
        directory,
        target_size = (150, 150),
        batch_size = batch_size,
        class_mode = "binary"
    )
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size:(i+1) * batch_size] = features_batch
        labels[i * batch_size:(i+1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    
    return features, labels

def features_extract(conv_base):
    train_features, train_labels = extract_features(conv_base, train_dir, 2000, batch_size = 20)
    validation_features, validation_labels = extract_features(conv_base, validation_dir, 1000, batch_size = 20)
    test_features, test_labels = extract_features(conv_base, test_dir, 1000, batch_size = 20)
    train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
    results = {
        "train": {
            "features": train_features,
            "labels": train_labels,
        },
        "validation": {
            "features": validation_features,
            "labels": validation_labels,
        },
        "test": {
            "features": test_features,
            "labels": test_labels,
        },
    }

    return results


def model_builder_without_data_augmentation():
    """不使用数据增强的 CNN 模型"""
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation = "relu"))
    model.add(layers.Dense(1, activation = "sigmoid"))

    return model


def model_builder_with_data_augmentation():
    """使用数据增强的 CNN 模型"""
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation = "relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation = "relu"))
    model.add(layers.Dense(1, activation = "sigmoid"))

    return model


def conv_base_builder():
    """pretrained network VGG16"""
    conv_base = VGG16(weights = "imagenet",
                      include_top = False,
                      input_shape = (150, 150, 3))
    
    return conv_base


def dnn_model_builder():
    """不使用数据增强的快速特征提取"""
    model = models.Sequential()
    model.add(layers.Dense(256, activation = "relu", input_dim = 4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation = "sigmoid"))

    return model


def conv_base_dnn_model_builder(conv_base):
    """使用数据增强的特征提取"""
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation = "relu"))
    model.add(layers.Dense(1, actication = "sigmoid"))

    return model


def model_compiler(model):
    model.compile(
        optimizer = optimizers.RMSprop(lr = 2e-5),
        loss = "binary_crossentropy",
        metrics = ['acc']
    )
    
    return model





# conv_base.trainable = True
# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name == "block5_conv1":
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else: 
#         layxr.trainable = False







def model_training(model, train_generator, validation_generator, epochs):
    """利用批量生成器拟合模型"""
    history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = epochs,
        validation_data = validation_generator,
        validation_steps = 50
    )

    return model, history


def model_training_extract_features(model, train_features, train_labels, validation_features, validation_labels):
    history = model.fit(
        train_features, train_labels,
        epochs = 30,
        batch_size = 20,
        validation_data  = (validation_features, validation_labels)
    )

    return model, history


def model_saver(model, model_path, model_name):
    model_path = os.path.join(model_path, model_name)
    model.save(model_path)


def model_visualer(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, "bo", label = "Training acc")
    plt.plot(epochs, val_acc, "b", label = "Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, "bo", label = "Train loss")
    plt.plot(epochs, val_loss, "b", label = "Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.show()


def data_augmentation():
    """
    1.模型过拟合处理--数据增强(data augmentation)
    2.显示几个随机增强后的训练图像
    """
    datagen = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2, 
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = "nearest"
    )
    fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
    img_path = fnames[3]
    img = image.load_img(img_path, target_size = (150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size = 1):
        plt.figure(i)
        imgplot = plt.imshow(image.array_to_img(batch[0]))
        i += 1
        if i % 4 == 0:
            break

    plt.show()


if __name__ == "__main__":
    # --------------------
    # data
    # --------------------
    data_class = ['train', 'validation', 'test']
    animal_class = ["cat", "dog"]
    # split_data(original_dataset_dir, small_data_dir, animal_class)
    PrintData(small_data_dir, data_class, animal_class)
    # --------------------
    # model
    # --------------------
    model_1_tag, model_2_tag, model_3_tag, model_4_tag = [1, 0, 0, 0]
    if model_1_tag == 1:
        # --------------------
        # model 1
        # --------------------
        model = model_builder_without_data_augmentation()
        model.summary()
        model = model_compiler(model)
        train_generator, validation_generator = data_processinger(model, is_data_augmentation = 0)
        model, history = model_training(model, train_generator, validation_generator, epochs = 30)
        model_saver(model, models_path, model_name = "cats_and_dogs_small_1.h5")
        model_visualer(history)
    elif model_2_tag == 1:
        # --------------------
        # model 2
        # --------------------
        model = model_builder_with_data_augmentation()
        model.summary()
        model = model_compiler(model)
        train_generator, validation_generator = data_processinger(model, is_data_augmentation = 1)
        model, history = model_training(model, train_generator, validation_generator, epochs = 30)
        model_saver(model, models_path, model_name = "cats_and_dogs_small_2.h5")
        model_visualer(history)
    elif model_3_tag == 1:
        # --------------------
        # model 3
        # --------------------
        conv_base = conv_base_builder()
        conv_base.summary()
        model = dnn_model_builder()
        model.summary()
        result = features_extract(conv_base)
        model = model_compiler(model)
        model, history = model_training_extract_features(
            model, 
            result["train"]["features"], result["train"]["labels"], 
            result["validation"]["features"], result["validation"]["labels"]
        )
        model_saver(model, models_path, model_name = "cats_and_dogs_small_3.h5")
        model_visualer(history)
    elif model_4_tag == 1:
        # --------------------
        # model 4
        # --------------------
        conv_base = conv_base_builder()
        conv_base.summary()
        model = conv_base_dnn_model_builder(conv_base)
        model.summary()
        train_generator, validation_generator = data_processinger(model, is_data_augmentation = 1)
        model = model_compiler(model)
        model, history = model_training(model, train_generator, validation_generator, epochs = 30)
        model_saver(model, models_path, model_name = "cats_and_dogs_small_4.h5")
        model_visualer(history)
    
    data_augmentation()
