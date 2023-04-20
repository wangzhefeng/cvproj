# -*- coding: utf-8 -*-


# ***************************************************
# * File        : model_training.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-03-19
# * Version     : 0.1.031923
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys
import tensorflow as tf
from tensorflow import keras
from config.config_loader import settings


def model_training(model, data_loader):
    """
    模型训练

    :param model: _description_
    :type model: _type_
    :param data_loader: _description_
    :type data_loader: _type_
    :return: _description_
    :rtype: _type_
    """
    # 模型编译
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = settings["MODEL"]["learning_rate"]),
        # optimizer = "adam",
        # optimizer = "rmsprop",
        # optimizer = tf.keras.optimizers.Adadelta(),
        # optimizer = tf.keras.optimizers.RMSprop()

        loss = tf.keras.losses.sparse_categorical_crossentropy,
        # loss = "sparse_categorical_crossentropy",
        # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        # loss = tf.keras.losses.categorical_crossentropy,
        # loss = "categorical_crossentropy",
        
        metrics = [
            tf.keras.metrics.sparse_categorical_accuracy
        ]
        # metrics = ["accuracy"],
    )
    # 模型训练
    history = model.fit(
        data_loader.train_data, 
        data_loader.train_label, 
        epochs = settings["MODEL"]["epochs"], 
        batch_size = settings["MODEL"]["batch_size"],
        verbose = 1,
        vadidation_data = (data_loader.test_data, data_loader.test_label),
        # validation_split = 0.2,
    )

    return model


def model_training_MLP(model, data_loader):
    num_epochs = settings["MODEL"]["epochs"]  # 5
    batch_size = settings["MODEL"]["batch_size"]  # 50
    learning_rate = settings["MODEL"]["learning_rate"]  # 1e-4

    @tf.function
    def train_one_step(X, y, optimizer):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true = y, y_pred = y_pred))
            # loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true = tf.one_hot(y, depth = tf.shape(y_pred)[-1]), y_pred = y_pred))
            tf.print("batch %d: loss %f" % (batch_index, loss.numpy()))
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))
    
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate =  learning_rate)
    # batches
    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    # model training
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        train_one_step(X, y, optimizer)
    # metric
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.predict(data_loader.test_data[start_index:end_index])
        sparse_categorical_accuracy.update_state(
            y_true = data_loader.test_label[start_index:end_index], 
            y_pred = y_pred
        )
    print("test accuracy: %f" % sparse_categorical_accuracy.result())


def model_training_CNN(model, data_loader):
    num_epochs = settings["MODEL"]["epochs"]
    batch_size = settings["MODEL"]["batch_size"]
    learning_rate = settings["MODEL"]["learning_rate"]

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate =  learning_rate)
    # batches
    num_batches = int(data_loader.num_train_data // batch_size * num_epochs)
    # model training
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(batch_size)
        print("X.shape:", X.shape)
        print("y.shape:", y.shape)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true = y, y_pred = y_pred))
            # loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true = tf.one_hot(y, depth = tf.shape(y_pred)[-1]), y_pred = y_pred))
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))

    # metric
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
        y_pred = model.predict(data_loader.test_data[start_index:end_index])
        sparse_categorical_accuracy.update_state(
            y_true = data_loader.test_label[start_index:end_index], 
            y_pred = y_pred
        )
    print("test accuracy: %f" % sparse_categorical_accuracy.result())


def model_training_temp(model, data_loader):
    train_ds = data_loader.data_shuffle(shuffle = 10000, shuffle_batch = 32)
    test_ds = data_loader.data_shuffle(shuffle = 10000, shuffle_batch = 32)
    # --------------------------------------
    # model compile
    # --------------------------------------
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name = "train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name = "train_accuracy")
    test_loss = tf.keras.metrics.Mean(name = "test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalCrossentropy(name = "test_accuracy")
    # --------------------------------------
    # model training
    # --------------------------------------
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    EPOCHS = 5
    for epoch in range(EPOCHS):
        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
        print(template.format(epoch + 1,
                            train_loss.result(),
                            train_accuracy.result() * 100,
                            test_loss.result(),
                            test_accuracy.result() * 100))
        # reset the metrics for the next epoch
        # train_loss.reset_states()
        # train_accuracy.reset_states()
        # test_loss.reset_states()
        # test_accuracy.reset_statess()



# 测试代码 main 函数
def main():
    # 1.数据加载
    from data_loader import data_loader
    # 2.数据预处理
    # 3.模型构建
    # 4.模型训练
    
    


if __name__ == "__main__":
    main()

