import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, params):
    """
    Regularized Residual GAF CNN - An improved version of the base residual D-GAF CNN.
    
    Enhancements:
    - Enabled residual connections
    - Added L2 regularization
    - Increased dropout rate
    - Maintained effective architecture components
    """
    
    # Defensive cast
    input_shape = tuple(input_shape)
    
    # --- Parameters with defaults ---
    filter_base = params.get("filter_base", 32)
    kernel_size = params.get("kernel_size", 3)
    dropout_rate = params.get("dropout_rate", 0.7)  # Increased from 0.5
    use_batch_norm = params.get("use_batch_norm", True)
    use_residual = params.get("use_residual", True)  # Now enabled
    l2_reg = params.get("l2_reg", 1e-4)  # Added L2 regularization
    activation = params.get("activation", "relu")
    
    # Regularization
    kernel_regularizer = keras.regularizers.l2(l2_reg)
    
    # Input
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Initial Conv Layer
    x = layers.Conv2D(
        filter_base, 
        kernel_size, 
        padding="same",
        kernel_regularizer=kernel_regularizer
    )(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    
    # Conv Blocks with Residual Connections
    filter_counts = [filter_base * (2 ** i) for i in range(4)]  # [32, 64, 128, 256]
    
    for filters in filter_counts:
        # First conv in block
        shortcut = x
        x = layers.Conv2D(
            filters, 
            kernel_size, 
            padding="same",
            kernel_regularizer=kernel_regularizer
        )(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        
        # Second conv in block
        x = layers.Conv2D(
            filters, 
            kernel_size, 
            padding="same",
            kernel_regularizer=kernel_regularizer
        )(x)
        if use_batch_norm:
            x = layers.BatchNormalization()(x)
        
        # Residual connection
        if use_residual:
            # Match dimensions if needed
            if shortcut.shape[-1] != x.shape[-1]:
                shortcut = layers.Conv2D(
                    filters, 
                    1, 
                    padding="same",
                    kernel_regularizer=kernel_regularizer
                )(shortcut)
            x = layers.Add()([x, shortcut])
        
        x = layers.Activation(activation)(x)
        x = layers.MaxPooling2D(2)(x)
    
    # Global Average Pooling instead of Flatten
    x = layers.GlobalAveragePooling2D()(x)
    
    # Regularization
    x = layers.Dropout(dropout_rate)(x)
    
    # Output Layer
    outputs = layers.Dense(
        num_classes, 
        activation="softmax",
        kernel_regularizer=kernel_regularizer
    )(x)
    
    model = keras.Model(inputs, outputs, name="regularized_residual_gaf_cnn")
    return model