import numpy as np
import tensorflow as tf
from tensorflow import keras

def build_model(input_shape, num_classes, params):
    """
    Enhanced 1D CNN Architecture based on insights from recent trials.
    
    Key Improvements:
    - Fixed 6 convolutional layers with kernel size 7
    - Increased filter count (64 base filters)
    - High dropout rates (0.8-0.85) for regularization
    - Global Average Pooling followed by dense layers
    - Extended training protocol support through early stopping and learning rate scheduling
    """
    
    # Define fixed architecture parameters based on insights
    num_conv_layers = 6
    kernel_size = 7
    base_filters = 64
    dense_units = 64
    dropout_rate = params.get("dropout_rate", 0.8)
    learning_rate = params.get("learning_rate", 0.0001)
    l2_reg = params.get("l2_regularization", 0.001)
    
    # Input layer
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Data Augmentation - Gaussian Noise for 1D signals
    if params.get("use_data_augmentation", False):
        x = keras.layers.GaussianNoise(0.1)(x)
    
    # Convolutional layers with residual connections
    for i in range(num_conv_layers):
        # Calculate number of filters for this layer (doubling pattern)
        num_filters = min(512, base_filters * (2 ** (i // 2)))
        
        # Residual connection
        shortcut = x
        
        # Convolutional block
        x = keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Add residual connection
        if shortcut.shape[-1] != num_filters:
            shortcut = keras.layers.Conv1D(num_filters, 1, padding="same")(shortcut)
        x = keras.layers.Add()([x, shortcut])
        
        # Max pooling after every two conv layers to reduce dimensionality
        if (i + 1) % 2 == 0:
            x = keras.layers.MaxPooling1D(2)(x)
            
    # Global Average Pooling instead of Flatten to reduce parameters
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense classification head
    x = keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    
    # Output layer
    x = keras.layers.Dense(num_classes)(x)
    outputs = keras.layers.Activation("softmax", dtype="float32")(x)
    
    # Create model
    model = keras.Model(inputs, outputs)
    
    # Compile with specified learning rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model