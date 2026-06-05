import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_model(input_shape, num_classes, params):
    """
    2D CNN Architecture optimized for GAF (Gramian Angular Field) images from SERS spectra.
    
    This model is designed based on insights from 73 stagnant 1D_CNN trials, shifting to 2D
    representation to escape performance plateau. The architecture leverages:
    
    - Deeper convolutional stacks (7-8 layers) with moderate filter sizes (32-64)
    - Larger kernel sizes (5-7) in early layers for broader receptive fields
    - Strategic use of batch normalization for stable training
    - Moderate dropout (~0.1-0.25) to prevent overfitting
    - Learning rates in the range 1e-4 to 1e-3
    
    The model processes 2D GAF images of shape (H, W, 1) where H=W (square images).
    
    Parameters:
    - input_shape: Tuple of (height, width, channels) for GAF images
    - num_classes: Number of output classes
    - params: Dictionary of hyperparameters (see below for expected keys)
    
    Expected params keys:
    - num_conv_layers (default: 7)
    - filters (default: 32)
    - kernel_size (default: 5)
    - dropout_rate (default: 0.2)
    - learning_rate (default: 5e-4)
    - weight_decay (default: 1e-4)
    - dense_units (default: 256)
    - use_batch_norm (default: True)
    - use_residual (default: True)
    """
    
    # --- Hyperparameters with defaults ---
    num_conv_layers = params.get("num_conv_layers", 7)
    base_filters = params.get("filters", 32)
    kernel_size = params.get("kernel_size", 5)
    dropout_rate = params.get("dropout_rate", 0.2)
    learning_rate = params.get("learning_rate", 5e-4)
    weight_decay = params.get("weight_decay", 1e-4)
    dense_units = params.get("dense_units", 256)
    use_batch_norm = params.get("use_batch_norm", True)
    use_residual = params.get("use_residual", True)
    
    # --- Regularizer ---
    reg = keras.regularizers.l2(weight_decay) if weight_decay > 0 else None
    
    # --- Model Definition ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # --- Convolutional Stack ---
    for i in range(num_conv_layers):
        # Calculate filters with moderate growth
        num_filters = min(512, base_filters * (2 ** (i // 2)))  # Grow filters more slowly
        
        # Residual Connection
        shortcut = x
        
        # Convolutional Layer
        x = layers.Conv2D(
            num_filters,
            kernel_size,
            padding="same",
            kernel_regularizer=reg
        )(x)
        
        # Batch Normalization
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
            
        # Activation
        x = layers.Activation("relu")(x)
        
        # Residual Connection
        if use_residual:
            # Match dimensions if needed
            if shortcut.shape[-1] != num_filters:
                shortcut = layers.Conv2D(num_filters, 1, padding="same", kernel_regularizer=reg)(shortcut)
            x = layers.Add()([x, shortcut])
            
        # Pooling (reduce spatial dimensions)
        x = layers.MaxPooling2D(2)(x)
        
        # Optional: Reduce kernel size in deeper layers
        if i >= 3:
            kernel_size = max(3, kernel_size - 1)
    
    # --- Global Pooling ---
    x = layers.GlobalAveragePooling2D()(x)
    
    # --- Dense Classifier ---
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    
    # --- Model Compilation ---
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model