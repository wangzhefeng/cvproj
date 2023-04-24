# -*- coding: utf-8 -*-

# ***************************************************
# * File        : MLP_tf_train_Checkpoint.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-04-24
# * Version     : 0.1.042422
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import argparse

import numpy as np
import tensorflow as tf

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# 模型超参数
parser = argparse.ArgumentParser(description = "Process some integers.")
parser.add_argument("--mode", default = "train", help = "train or test")
parser.add_argument("--num_epochs", default = 1)
parser.add_argument("--batch_size", default = 50)
parser.add_argument("--learning_rate", default = 0.001)
args = parser.parse_args()


def train(root_path, model, data_loader):
    # Model build
    optimizer = tf.keras.optimizers.Adam(learning_rate = args.learning_rate)
    num_batches = int(data_loader.num_train_data // args.batch_size * args.num_epochs)
    # Checkpoint
    checkpoint = tf.train.Checkpoint(myAwesomeModel = model)
    manager = tf.train.CheckpointManager(
        checkpoint, 
        directory = os.path.join(root_path, "save"), 
        checkpoint_name = "model.ckpt", 
        max_to_keep = 10
    )
    # Model training
    for batch_index in range(1, num_batches + 1):
        X, y = data_loader.get_batch(args.batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true = y, y_pred = y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))
        if batch_index % 100 == 0:
            # version 1
            # path = checkpoint.save(os.path.join(root_path, "save/model.ckpt"))
            # version 2
            path = manager.save(checkpoint_number = batch_index)
            print("model saved to %s" % path)


def test(root_path, model, data_loader):
    # Checkpoint
    checkpoint = tf.train.Checkpoint(myAwesomeModel = model)
    checkpoint.restore(
        tf.train.latest_checkpoint(os.path.join(root_path, "save"))
    )
    y_pred = np.argmax(model.predict(data_loader.test_data), axis = -1)
    print(f"test accuracy: {sum(y_pred == data_loader.test_label) / data_loader.num_test_data}")





# 测试代码 main 函数
def main():
    # model
    if args.mode == "train":
        train()
    if args.mode == "test":
        test()

if __name__ == "__main__":
    main()
