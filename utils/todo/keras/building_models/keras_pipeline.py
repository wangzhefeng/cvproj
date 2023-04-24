# -*- coding: utf-8 -*-


# ***************************************************
# * File        : keras_pipeline.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-02-24
# * Version     : 0.1.022400
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # or any {"0", "1", "2"}

# import logging
# logging.getLogger("tensorflow").disabled = True

# import tensorflow as tf
# tf.get_logger().setLevel("INFO")

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ====================================================
# Keras functional-api
# ====================================================
# ------------------------------------------
# data
# ------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# ------------------------------------------
# build model
# ------------------------------------------
# RGB images inputs
inputs = keras.Input(shape = (28, 28))

# center-crop images to 150x150
x = keras.layers.CenterCrop(height = 28, width = 28)(inputs)
# rescale images to [0, 1]
x = keras.layers.Rescaling(1.0 / 255)(x)
# TODO
x = keras.layers.Flatten()(x)
# apply some convolution and pooling layers
x = keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu")(x)
x = keras.layers.MaxPooling2D(pool_size = (3, 3))(x)
x = keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu")(x)
x = keras.layers.MaxPooling2D(pool_size = (3, 3))(x)
x = keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu")(x)

# Apply global average pooling to get flat feature vectors
x = keras.layers.GlobalAveragePooling2D()(x)

# add a dense classifier on top
num_classes = 10
outputs = keras.layers.Dense(num_classes, activation = "softmax")(x)

# instantiate a Model object
model = keras.Model(inputs = inputs, outputs = outputs, name = "wangzf")
print(model.summary())

# apply model to data
# data = np.random.randint(0, 256, size = (64, 200, 200, 3)).astype("float32")
# processed_data = model(data)
# print(processed_data.shape)


# ------------------------------------------
# model compile
# ------------------------------------------
model.compile(
    optimizer = keras.optimizers.RMSprop(learning_rate = 1e-3),
    loss = keras.losses.CategoricalCrossentropy(),
    metrics = [
        keras.metrics.CategoricalAccuracy(name = "acc"),
    ]
)
model.compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
)


# ------------------------------------------
# model fit-Numpy
# ------------------------------------------
batch_size = 64
print("Fit on Numpy data.")
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = 1)
print(history.history)

# ------------------------------------------
# model fit-Dataset
# ------------------------------------------
batch_size = 64
print("Fit on Dataset.")
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
history = model.fit(dataset, epochs = 1)
print(history.history)

# ------------------------------------------
# model validate
# ------------------------------------------
batch_size = 64
print("Fit on Validate dataset.")
val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
history = model.fit(dataset, epochs = 1, validation_data = val_dataset)
print(history.history)

# ------------------------------------------
# callbacks for checkpointing
# ------------------------------------------
# callbacks = [
#     keras.callbacks.ModelCheckpoint(
#         filepath = "path/to/my/model_{epoch}",
#         save_freq = "epoch"
#     )
# ]
# model.fit(dataset, epochs = 2, callbacks = callbacks)

# ------------------------------------------
# Tensorboard
# ------------------------------------------
# callbacks = [
#     keras.callbacks.TensorBoard(log_dir = "./logs")
# ]
# model.fit(dataset, epochs = 2, callbacks = callbacks)

# tensorboard --logdir=./logs

# ------------------------------------------
# model evaluate
# ------------------------------------------
loss, acc = model.evaluate(val_dataset)
print("loss: %.2f" % loss)
print("acc: %.2f" % acc)

# or
# predictions = model.predict(val_dataset)
# print(predictions.shape)


# ------------------------------------------
# 自定义训练步骤
# ------------------------------------------
class CustomModel(keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            # Compute the loss value
            loss = self.compiled_loss(y, y_pred, regularization_losses = self.losses)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

# Construct and compile an instance of CustomModel
inputs = keras.Input(shape = (32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer = 'adam', loss = 'mse', metrics = [...])

# Just use `fit` as usual
model.fit(dataset, epochs = 3, callbacks = ...)

# ------------------------------------------
# 调试模型模式
# ------------------------------------------
model.compile(optimizer = "adam", loss = "mse", run_eagerly = True)

# ------------------------------------------
# GPU 分布式训练
# ------------------------------------------
# 创建一个 MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

# 开启一个 strategy scope
with strategy.scope():
    model = keras.Model()
    model.compile()

train_dataset, val_dataset, test_dataset = get_dataset()
# 在所有可用的设备上训练模型
model.fit(train_dataset, epochs = 2, validation_data = val_dataset)
# 在所有可用的设备上测试模型
model.evaluate(test_dataset)



# ------------------------------------------
# 异步处理
# ------------------------------------------
samples = np.array([
    ["This is the 1st sample."], 
    ["And here's the 2nd sample."]
])
labels = [
    [0], 
    [1]
]
# Prepare a TextVectorization layer.
vectorizer = keras.layers.TextVectorization(output_mode = "int")
vectorizer.adapt(samples)

# Asynchronous preprocessing: the text vectorization is part of the tf.data pipeline.
# First, create a dataset
dataset = tf.data.Dataset.from_tensor_slices((samples, labels)).batch(2)
# Apply text vectorization to the samples
dataset = dataset.map(lambda x, y: (vectorizer(x), y))
# Prefetch with a buffer size of 2 batches
dataset = dataset.prefetch(2)

# Our model should expect sequences of integers as inputs
inputs = keras.Input(shape=(None,), dtype="int64")
x = keras.layers.Embedding(input_dim=10, output_dim=32)(inputs)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse", run_eagerly=True)
model.fit(dataset)


# ------------------------------------------
# 超参数搜索调优
# ------------------------------------------
def build_model(hp):
    """返回已编译的模型

    Args:
        hp ([type]): [description]

    Returns:
        [type]: [description]
    """
    inputs = keras.Input(shape = (784,))
    x = keras.layers.Dense(
        units = hp.Int("units", min_value = 32, max_value = 512, step = 32),
        activation = "relu",
    )(inputs)
    outputs = keras.layers.Dense(10, activation = "softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer = keras.optimizers.Adam(hp.Choice("leraning_rate", values = [1e-2, 1e-3, 1e-4])),
        loss = "sparse_categorical_crossentropy",
        metrics = ["accuracy"]
    )

    return model

tuner = kerastuner.tuners.Hyperband(
    build_model,
    objective = "val_loss",
    max_epochs = 100,
    max_trials = 200,
    executions_per_trial = 2,
    directory = "my_dir"
)
tuner.search(dataset, validation_data = val_dataset)
models = tuner.get_best_models(num_models = 2)
tuner.results_summary()
