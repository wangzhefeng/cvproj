import tensorflow as tf
import os

num_epochs = 10
batch_size = 32
learning_rate = 0.001

# root
root_dir = "/Users/zfwang/project/machinelearning/deeplearning"
# project
project_path = os.path.join(root_dir, "deeplearning/src/project_src/cats_vs_dogs")
# model save
models_path = os.path.join(project_path, "models")
# data
data_dir = os.path.join(root_dir, "datasets/cats_vs_dogs/cats_and_dogs_small")
train_dir = os.path.join(data_dir, "train")
train_cats_dir = os.path.join(train_dir, "cat")
train_dogs_dir = os.path.join(train_dir, "dog")
# validation_dir = os.path.join(data_dir, "validation")
# validation_cats_dir = os.path.join(validation_dir, "cat")
# validation_dogs_dir = os.path.join(validation_dir, "dogs")
# test_dir = os.path.join(data_dir, "test")
# test_cats_dir = os.path.join(test_dir, "cat")
# test_dogs_dir = os.path.join(test_dir, "dog")
test_dir = os.path.join(data_dir, "test")
test_cats_dir = os.path.join(test_dir, "cat")
test_dogs_dir = os.path.join(test_dir, "dog")


def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0
    return image_resized, label


def get_train_dataset():
    """
    构建训练数据集
    """
    train_cat_filenames = tf.constant([os.path.join(train_cats_dir, filename) for filename in os.listdir(train_cats_dir)])
    train_dog_filenames = tf.constant([os.path.join(train_dogs_dir, filename) for filename in os.listdir(train_dogs_dir)])
    train_filenames = tf.concat([train_cat_filenames, train_dog_filenames], axis = -1)
    train_labels = tf.concat([tf.zeros(train_cat_filenames.shape, dtype = tf.int32), tf.zeros(train_dog_filenames.shape, dtype = tf.int32)], axis = -1)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(map_func = _decode_and_resize, num_parallel_calls = 8)
    train_dataset = train_dataset.shuffle(buffer_size = 23000)
    train_dataset = train_dataset.batch(batch_size = batch_size)
    train_dataset = train_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    return train_dataset


def get_test_dataset():
    """
    构建测试数据集
    """
    test_cat_filenames = tf.constant([os.path.join(test_cats_dir, filename) for filename in os.listdir(test_cats_dir)])
    test_dog_filenames = tf.constant([os.path.join(test_dogs_dir, filename) for filename in os.listdir(test_dogs_dir)])
    test_filenames = tf.concat([test_cat_filenames, test_dog_filenames], axis = -1)
    test_labels = tf.concat([tf.zeros(test_cat_filenames.shape, dtype = tf.int32), tf.zeros(test_dog_filenames.shape, dtype = tf.int32)], axis = -1)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
    test_dataset = test_dataset.map(map_func = _decode_and_resize)
    test_dataset = test_dataset.batch(batch_size = batch_size)
    return test_dataset


def build_model():
    """
    构建模型
    """
    # Sequential
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(32, 3, activation = "relu", input_shape = (256, 256, 3)),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Conv2D(32, 5, activation = "relu"),
    #     tf.keras.layers.MaxPooling2D(),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(64, activation = "relu"),
    #     tf.keras.layers.Dense(2, activation = "softmax")
    # ])
    # Functional API
    inputs = tf.keras.Input(shape = (256, 256, 3))
    conv1 = tf.keras.layers.Conv2D(32, 3, activation = tf.nn.relu)(inputs)
    max1 = tf.keras.layers.MaxPooling2D()(conv1)
    conv2 = tf.keras.layers.Conv2D(32, 5, activation = tf.nn.relu)(max1)
    max2 = tf.keras.layers.MaxPooling2D()(conv2)
    flatten = tf.keras.layers.Flatten()(max2)
    dense1 = tf.keras.layers.Dense(64, activation = tf.nn.relu)(flatten)
    outputs = tf.keras.layers.Dense(2, activation = tf.nn.softmax)(dense1)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model




if __name__ == "__main__":
    # 数据集
    train_dataset = get_train_dataset()
    test_dataset = get_test_dataset()
    # 构建模型
    model = build_model()
    # 模型编译
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
        loss = tf.keras.losses.sparse_categorical_crossentropy,
        metrics = [tf.keras.metrics.sparse_categorical_accuracy]
    )
    # 模型训练
    model.fit(train_dataset, epochs = num_epochs)
    # 模型测试
    print(model.metrics_names)
    print(model.evaluate(test_dataset))
