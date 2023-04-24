# -*- coding: utf-8 -*-

# ***************************************************
# * File        : advanced.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2022-04-05
# * Version     : 0.1.040521
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import tensorflow as tf

# data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# model
class MyModel(tf.keras.Model):

    def __init__(self) -> None:
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation = "relu")
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation = "relu")
        self.d2 = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()

# loss
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
# optimizer
optimizer = tf.keras.optimizers.Adam()
# metrics
train_loss = tf.keras.metrics.Mean(name = "train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = "train_accuracy")
test_loss = tf.keras.metrics.Mean(name = "test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = "test_accuracy")


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training = True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, training = False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    
    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {train_loss.result()}, "
        f"Accuracy: {train_accuracy.result() * 100}, "
        f"Test Loss: {test_loss.result()}, "
        f"Test Accuracy: {test_accuracy.result() * 100}"
    )
"""
Epoch 1, Loss: 0.13121572136878967Accuracy: 95.99666595458984Test Loss: 0.0608455054461956Test Accuracy: 98.00999450683594
Epoch 2, Loss: 0.0411057285964489Accuracy: 98.69999694824219Test Loss: 0.054041847586631775Test Accuracy: 98.30999755859375
Epoch 3, Loss: 0.022492099553346634Accuracy: 99.25333404541016Test Loss: 0.06425648182630539Test Accuracy: 98.06999969482422
Epoch 4, Loss: 0.01375863142311573Accuracy: 99.52832794189453Test Loss: 0.06595206260681152Test Accuracy: 98.12999725341797
Epoch 5, Loss: 0.00915294885635376Accuracy: 99.70166778564453Test Loss: 0.065868079662323Test Accuracy: 98.29000091552734
"""



print(model.summary())
"""
Model: "my_model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             multiple                  320       
                                                                 
 flatten (Flatten)           multiple                  0         
                                                                 
 dense (Dense)               multiple                  2769024   
                                                                 
 dense_1 (Dense)             multiple                  1290      
                                                                 
=================================================================
Total params: 2,770,634
Trainable params: 2,770,634
Non-trainable params: 0
_________________________________________________________________
None
"""



# 测试代码 main 函数
def main():
    pass


if __name__ == "__main__":
    main()

