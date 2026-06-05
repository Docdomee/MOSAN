import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_model(input_shape, num_classes, params):
    """
    Optimized 2D GAF ResNet Architecture incorporating key success patterns:
    - 5-7 layer deep ResNet with 64-128 filters
    - 3x3 kernels exclusively
    - Batch normalization after each conv
    - Moderate dropout (0.3-0.4)
    - L2 regularization (~1e-5)
    - Global average pooling before dense layers
    - Residual connections for gradient flow
    """
    
    # --- Hyperparameters from params or defaults ---
    num_conv_layers = params.get("num_conv_layers", 6)  # Target 5-7 layers
    base_filters = params.get("filters", 64)  # Start with 64 filters
    kernel_size = params.get("kernel_size", 3)
    dropout_rate = params.get("dropout_rate", 0.35)  # Moderate dropout
    learning_rate = params.get("learning_rate", 0.001)
    weight_decay = params.get("weight_decay", 1e-5)  # L2 regularization
    use_residual = params.get("use_residual", True)  # Enable residuals
    
    reg = keras.regularizers.l2(weight_decay)

    def residual_block(x, filters, stride=1):
        """Standard ResNet residual block with 2 conv layers"""
        shortcut = x
        
        # First conv layer
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same", 
                          kernel_regularizer=reg, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Second conv layer
        x = layers.Conv2D(filters, kernel_size, strides=1, padding="same", 
                          kernel_regularizer=reg, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if needed (dimension mismatch)
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", 
                                     kernel_regularizer=reg, use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Add residual connection
        if use_residual:
            x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        return x

    def residual_block_no_stride(x, filters):
        """Residual block without stride (for same dimensions)"""
        return residual_block(x, filters, stride=1)

    # --- Model Definition ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # --- Data Augmentation (2D) ---
    if params.get("use_data_augmentation"):
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)

    # Initial conv layer
    x = layers.Conv2D(base_filters, kernel_size, strides=1, padding="same", 
                      kernel_regularizer=reg, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # ResNet-like blocks with increasing filters
    for i in range(num_conv_layers):
        filters = base_filters * (2 ** (i // 2))  # Double filters every 2 layers
        stride = 2 if i > 0 and i % 2 == 1 else 1  # Downsample every other block
        
        if i == 0:
            x = residual_block(x, filters, stride=stride)
        else:
            x = residual_block_no_stride(x, filters)
        
        # Additional block at same resolution
        x = residual_block_no_stride(x, filters)
    
    # Global Average Pooling instead of Flatten
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layer with regularization
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    
    # --- Compile Model ---
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model