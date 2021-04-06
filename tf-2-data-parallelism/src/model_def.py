import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50


def get_resnet50(**kwargs):
    default_settings = {'transfer_learning': False}
    default_settings.update(kwargs)

    if not default_settings['transfer_learning']:
        model_weights = None
    else:
        model_weights = 'imagenet'

    conv_base = ResNet50(weights='imagenet', include_top=False,
                         input_shape=(32, 32, 3)
                         )
    model = tf.keras.Sequential()
    model.add(conv_base)  # Adds the base model
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    # Add the Dense layers along with activation and batch normalization
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, activation='softmax'))  # This is the classification layer

    return model
