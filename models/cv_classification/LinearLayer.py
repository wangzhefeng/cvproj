import numpy as np
import tensorflow as tf


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super.__init__()
        self.units = units
    
    def build(self, input_shape):
        self.w = self.add_variable(
            name = "w", 
            shape = [input_shape[-1], self.units], 
            initializer = tf.zeros_initializer()
        )
        self.b = self.add_variable(
            name = "b",
            shape = [self.units],
            initializer = tf.zeros_initializer()
        )
    
    def call(self, inputs):
        y_pred = tf.matmul(inputs, self.w) + self.b
        return y_pred


class LinearModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer = LinearLayer(untis = 1)
    
    def call(self, inputs):
        output = self.layer(inputs)
        return output


if __name__ == "__main__":
    num_epochs = 100
    batch_size = np.nan
    learning_rate = 1e-3

    # data
    X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0], [20.0]])

    # model
    model = LinearModel()

    # optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate)
    
    # model training
    for i in range(num_epochs):
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.reduce_sum(tf.square(y_pred - y))
        grads = tape.gradient(loss, model.variables)

        optimizer.apply_gradients(grads_and_vars = zip(grads, model.variables))

    print(model.variables)
