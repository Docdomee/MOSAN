import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape, num_classes, params=None):
    """
    Fixed 2D_GENERIC_IMAGE architecture for CIFAR-10.
    Correctly accepts 'params' argument to satisfy training_worker interface.
    """
    if params is None:
        params = {}

    inputs = tf.keras.Input(shape=input_shape)
    
    # 1. Normalization (Handle 0-255 inputs if not already handled)
    x = layers.Rescaling(1./255)(inputs)
    
    # 2. Convolutional Blocks
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # 3. Dense / Classification
    x = layers.GlobalAveragePooling2D()(x)
    
    dense_units = params.get("dense_units", 64)
    dropout_rate = params.get("dropout_rate", 0.5)
    
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model
