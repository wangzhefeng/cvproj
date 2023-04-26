import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


num_epoch = 5
batch_size = 19
learning_rate = 1e-3


# data
dataset = tfds.load("tf_flowers", split = tfds.Split.TRAIN, as_supervised = True)
dataset = dataset.map(lambda img, label: (tf.image.resize(img, (224, 224)) / 255.0, label)) \
                 .shuffle(1024) \
                 .batch(batch_size)

# model
model = tf.keras.applications.MobileNetV2(weights = None, include_top = True, classes = 5)

# model training
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
for e in range(num_epoch):
    for images, labels in dataset:
        with tf.GradientTape() as tape:
            labels_pred = model(images, training = True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true = labels, y_pred = labels_pred))
            print("loss %f" % loss.numpy())
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars = zip(grads, model.trainable_variables))
        print(labels_pred)
        optimizer.apply_gradients(grads_and_vars = zip(grads, model.trainable_variables))

