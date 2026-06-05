import tensorflow as tf
from tensorflow import keras
import numpy as np

def build_model(input_shape, num_classes, params):
    """
    Improved 2D GAF model based on ResNet-inspired architecture.
    
    Design Improvements:
    - 4-6 convolutional layers with progressive filter scaling (32→64→128→256)
    - Residual connections for better gradient flow
    - Moderate kernel sizes (3-5)
    - 512-unit dense layer for classification
    - Global Average Pooling instead of Flatten
    """

    # --- Configuration ---
    reg = keras.regularizers.l2(params.get("l2_reg", 1e-4))
    num_conv_layers = params.get("num_conv_layers", 5)  # Target 4-6 layers
    base_filters = params.get("base_filters", 32)
    kernel_size = params.get("kernel_size", 3)  # Moderate kernel size
    dense_units = params.get("dense_units", 512)  # Increased dense units
    dropout_rate = params.get("dropout_rate", 0.5)
    
    # Ensure kernel size is within optimal range
    if kernel_size not in [3, 5]:
        kernel_size = 3

    # --- Model Definition ---
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # --- Data Augmentation (2D) ---
    if params.get("use_data_augmentation", False):
        x = keras.layers.RandomFlip("horizontal")(x)
        x = keras.layers.RandomRotation(0.1)(x)
        x = keras.layers.RandomZoom(0.1)(x)

    # --- Initial Convolution ---
    x = keras.layers.Conv2D(
        base_filters, 
        (kernel_size, kernel_size), 
        padding="same",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # --- Residual Blocks ---
    for i in range(num_conv_layers):
        num_filters = min(512, base_filters * (2 ** i))  # Progressive filter scaling
        
        # Residual block with 2 convolutions
        shortcut = x
        
        # First conv in block
        x = keras.layers.Conv2D(
            num_filters,
            (kernel_size, kernel_size),
            padding="same",
            kernel_regularizer=reg
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Second conv in block
        x = keras.layers.Conv2D(
            num_filters,
            (kernel_size, kernel_size),
            padding="same",
            kernel_regularizer=reg
        )(x)
        x = keras.layers.BatchNormalization()(x)
        
        # Adjust shortcut if needed
        if shortcut.shape[-1] != num_filters:
            shortcut = keras.layers.Conv2D(num_filters, (1, 1), padding="same")(shortcut)
        
        # Add residual connection
        x = keras.layers.Add()([x, shortcut])
        x = keras.layers.Activation("relu")(x)
        
        # Downsample except for last layer
        if i < num_conv_layers - 1:
            x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # --- Classifier ---
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(num_classes)(x)
    outputs = keras.layers.Activation("softmax", dtype="float32")(x)

    # --- Compile Model ---
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model