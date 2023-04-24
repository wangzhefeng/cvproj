import numpy as np
import tensorflow as tf
import MNISTLoader


"""
CNN
"""


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters = 32,           # 卷积层神经元(卷积核)的数目
            kernel_size = [5, 5],   # 感受野的大小
            padding = "same",       # padding 策略(vaild 或 same)
            activation = tf.nn.relu # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size = [2, 2], strides = 2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters = 64,
            kernel_size = [5, 5],
            padding =  "same",
            activation = tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size = [2, 2], strides = 2)
        self.flatten = tf.keras.layers.Reshape(target_shape = (7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units = 1024, activation = tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units = 10)
    
    def call(self, inputs):    # [batch_size, 28, 28, 32]
        print(inputs.shape)
        x = self.conv1(inputs) # [batch_size, 28, 28, 32]
        print(x.shape)
        x = self.pool1(x)     # [batch_size, 14, 14, 32]
        print(x.shape)
        x = self.conv2(x)     # [batch_size, 14, 14, 64]
        print(x.shape)
        x = self.pool2(x)     # [batch_size, 7, 7, 64]
        print(x.shape)
        x = self.flatten(x)   # [batch_size, 7 * 7 * 64]
        print(x.shape)
        x = self.dense1(x)    # [batch_size, 1024]
        x = self.dense2(x)    # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output



if __name__ == "__main__":
    num_epochs = 5
    batch_size = 50
    learning_rate = 1e-4
    
    # data
    data_loader = MNISTLoader.MNISTLoader()

    # model
    model = CNN()

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
