import tensorflow as tf
from tensorflow import keras
import numpy as np

def build_model(input_shape, num_classes, params):
    """
    Deep 2D CNN architecture specifically designed for GAF (Gramian Angular Field) images
    generated from SERS spectra data.

    This model follows the design strategy derived from the analysis of 1D_CNN trials:
    - Deep architecture (7+ conv layers)
    - Moderate filter counts with small kernels
    - Moderate dropout and low weight decay
    - Relatively higher learning rates
    - Batch normalization disabled based on successful trial patterns
    """
    
    # --- Input ---
    inputs = keras.Input(shape=input_shape)
    
    # --- Data Augmentation (2D) ---
    if params.get("use_data_augmentation", False):
        x = keras.layers.RandomFlip("horizontal")(inputs)
        x = keras.layers.RandomRotation(0.1)(x)
        x = keras.layers.RandomZoom(0.1)(x)
    else:
        x = inputs
    # ------------------------------

    # --- Regularization ---
    reg = keras.regularizers.l2(params.get("weight_decay", 1e-6)) if params.get("weight_decay", 0) > 0 else None
    dropout_rate = params.get("dropout_rate", 0.2)
    # ----------------------

    # --- Convolutional Blocks ---
    # Block 1
    x = keras.layers.Conv2D(
        filters=params.get("filters", 32),
        kernel_size=params.get("kernel_size", 3),
        padding="same",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)

    # Block 2
    x = keras.layers.Conv2D(
        filters=params.get("filters", 32),
        kernel_size=params.get("kernel_size", 3),
        padding="same",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)

    # Block 3
    x = keras.layers.Conv2D(
        filters=params.get("filters", 32) * 2,  # 64
        kernel_size=params.get("kernel_size", 3),
        padding="same",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)

    # Block 4
    x = keras.layers.Conv2D(
        filters=params.get("filters", 32) * 2,  # 64
        kernel_size=params.get("kernel_size", 3),
        padding="same",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)

    # Block 5
    x = keras.layers.Conv2D(
        filters=params.get("filters", 32) * 4,  # 128
        kernel_size=3,  # Fixed to smaller kernel as per successful trials
        padding="same",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)

    # Block 6
    x = keras.layers.Conv2D(
        filters=params.get("filters", 32) * 4,  # 128
        kernel_size=3,  # Fixed to smaller kernel
        padding="same",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)

    # Block 7
    x = keras.layers.Conv2D(
        filters=params.get("filters", 32) * 4,  # 128
        kernel_size=3,  # Fixed to smaller kernel
        padding="same",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=2)(x)
    
    # Global Average Pooling to reduce spatial dimensions
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # --- Dense Layers ---
    x = keras.layers.Dense(
        units=params.get("dense_units", 256),
        activation="relu",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    
    # Output layer
    x = keras.layers.Dense(num_classes)(x)
    outputs = keras.layers.Activation("softmax", dtype="float32")(x)
    
    # --- Model Creation ---
    model = keras.Model(inputs, outputs)
    
    # --- Optimizer ---
    optimizer = keras.optimizers.Adam(
        learning_rate=params.get("learning_rate", 1e-3)
    )
    
    # --- Compilation ---
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model