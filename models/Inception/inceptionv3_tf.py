from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input


"""
Build InceptionV3 over a custom input tensor
"""

# this could also be the output a different Keras model or layer
input_tensor = Input(shape = (224, 224, 3))
model = InceptionV3(
    input_tensor = input_tensor, 
    weights = 'imagenet', 
    include_top = True
)

