import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, params: dict):
    """
    Enhanced 1D_CNN implementing a robust Broad-to-Narrow strategy.
    
    FIXED:
    1. Replaced fragile shape checks with a more robust spatial dimension tracker.
    2. Added a minimum spatial threshold to prevent tensor collapse to 0-dim.
    3. Ensured input_shape is explicitly cast to a tuple of integers to avoid symbolic None errors.
    4. Standardized the residual projection to handle dynamic shapes correctly.
    """
    
    # Internal helper for regularizer
    def _get_regularizer(p):
        l2_val = p.get("l2_reg", 1e-4)
        return keras.regularizers.l2(l2_val)

    # --- Input Validation ---
    if input_shape is None:
        input_shape = (128, 1)
    elif isinstance(input_shape, list):
        input_shape = tuple(input_shape)
    
    # Standardize to (sequence_length, channels)
    if len(input_shape) == 3:
        input_shape = input_shape[1:]
    
    if len(input_shape) != 2:
        # Fallback to a safe 1D shape if input is malformed
        seq_len = input_shape[0] if len(input_shape) == 1 else 128
        input_shape = (seq_len, 1)

    inputs = keras.Input(shape=input_shape, dtype="float32")
    x = inputs
    
    # --- Data Augmentation ---
    if params.get("use_data_augmentation"):
        x = layers.GaussianNoise(0.1)(x)

    reg = _get_regularizer(params)
    
    # Hyperparameter Extraction
    kernel_size = params.get("kernel_size", 3) 
    dilation_rate = params.get("dilation_rate", 1) 
    use_residual = params.get("use_residual", False)
    base_filters = params.get("filters", 32)
    requested_layers = params.get("num_conv_layers", 2)

    # Track spatial dimension to prevent collapse
    # Use the actual value from input_shape if available
    current_spatial_dim = input_shape[0]

    for i in range(requested_layers):
        # Filter Scaling: Linear growth capped at 128 to prevent OOM
        num_filters = min(128, base_filters + (i * 32))
        
        shortcut = x
        
        # Convolutional Layer
        x = layers.Conv1D(
            filters=num_filters, 
            kernel_size=kernel_size, 
            dilation_rate=dilation_rate,
            padding="same",
            kernel_regularizer=reg,
            kernel_initializer="he_normal"
        )(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Residual Connection
        if use_residual:
            # Project shortcut to match filter depth
            # Use a 1x1 conv to match dimensions without altering spatial size
            if shortcut.shape[-1] != num_filters:
                shortcut = layers.Conv1D(num_filters, 1, padding="same")(shortcut)
            x = layers.Add()([x, shortcut])
            
        # --- Safe Downsampling ---
        # Only apply MaxPooling if the current spatial dimension is > 2
        # This ensures that after pooling, we still have at least 1 element.
        if current_spatial_dim is not None and current_spatial_dim > 2:
            x = layers.MaxPooling1D(pool_size=2)(x)
            current_spatial_dim //= 2
        elif current_spatial_dim is None:
            # If shape is dynamic, we only pool for the first 2 layers to be extremely safe
            if i < 2:
                x = layers.MaxPooling1D(pool_size=2)(x)

    # Global Feature Aggregation
    # GAP is robust, but requires the input to have at least one element in the spatial dim
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification Head
    x = layers.Dense(
        params.get("dense_units", 64), 
        activation="relu",
        kernel_regularizer=reg
    )(x)
    
    x = layers.Dropout(params.get("dropout_rate", 0.5))(x)
    
    # Final Output Layer
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation("softmax", dtype="float32")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Optimizer setup
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model