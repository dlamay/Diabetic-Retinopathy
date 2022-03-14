from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow import keras

def resnet50(input_size):
    base_model = ResNet50(
        weights = 'imagenet',
        include_top = False
    )
    inputs = keras.Input(shape=input_size)
    x = base_model(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(5, activation = 'softmax')(x)

    return keras.Model(inputs, outputs)