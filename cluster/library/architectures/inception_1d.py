import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, params):
    """
    Inception-1D Architecture (Hackable Baseline).
    
    Key Idea: Multi-scale processing.
    Each block applies 1x1, 3x3, and 5x5 convolutions in parallel to capture patterns of different sizes.
    """
    
    dropout_rate = params.get("dropout_rate", 0.4)
    reg = keras.regularizers.l2(1e-4)

    def inception_module(x, filters=32):
        """
        Naive Inception Module for 1D.
        Parallel paths:
        1. 1x1 Conv
        2. 1x1 Conv -> 3x3 Conv
        3. 1x1 Conv -> 5x5 Conv
        4. MaxPool -> 1x1 Conv
        """
        # Path 1: 1x1
        p1 = layers.Conv1D(filters, 1, padding="same", activation="relu", kernel_regularizer=reg)(x)
        
        # Path 2: 3x3
        p2 = layers.Conv1D(filters, 1, padding="same", activation="relu", kernel_regularizer=reg)(x)
        p2 = layers.Conv1D(filters, 3, padding="same", activation="relu", kernel_regularizer=reg)(p2)
        
        # Path 3: 5x5
        p3 = layers.Conv1D(filters, 1, padding="same", activation="relu", kernel_regularizer=reg)(x)
        p3 = layers.Conv1D(filters, 5, padding="same", activation="relu", kernel_regularizer=reg)(p3)
        
        # Path 4: Pool
        p4 = layers.MaxPooling1D(3, strides=1, padding="same")(x)
        p4 = layers.Conv1D(filters, 1, padding="same", activation="relu", kernel_regularizer=reg)(p4)
        
        return layers.Concatenate()([p1, p2, p3, p4])

    # --- Build ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Stem
    x = layers.Conv1D(64, 7, strides=2, padding="same", activation="relu", kernel_regularizer=reg)(x)
    x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
    
    # Inception Blocks
    x = inception_module(x, filters=32)
    x = inception_module(x, filters=32)
    x = layers.MaxPooling1D(2)(x)
    
    x = inception_module(x, filters=64)
    x = inception_module(x, filters=64)
    x = layers.MaxPooling1D(2)(x)
    
    x = inception_module(x, filters=128)
    
    # Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    layers.Dense(num_classes)(x)`n    outputs = layers.Activation("softmax", dtype="float32")(x)
    
    model = keras.Model(inputs, outputs, name="Inception_1D")
    
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
