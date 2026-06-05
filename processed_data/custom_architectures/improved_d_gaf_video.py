import numpy as np
import tensorflow as tf
from tensorflow import keras

def build_model(input_shape, num_classes, params):
    """
    Improved 3D GAF Video CNN with residual connections, enhanced regularization,
    and optimized hyperparameters based on pattern analysis.
    
    Improvements:
    - Increased depth (6-8 layers)
    - Residual connections
    - Better regularization (BatchNorm + Dropout)
    - Smaller kernel sizes
    - Controlled filter growth
    """
    
    # Import inside function as required
    from tensorflow.keras import layers, regularizers, models
    
    def _get_regularizer(params):
        """Get L2 regularizer based on weight_decay parameter."""
        wd = params.get("weight_decay", 1e-5)
        return regularizers.l2(wd) if wd > 0 else None

    # Set default parameters aligned with pattern analysis
    reg = _get_regularizer(params)
    
    # Depth configuration - target 6-8 layers based on analysis
    requested_layers = params.get("num_conv_layers", 7)  # Default to 7 layers
    min_dim = min(input_shape[1], input_shape[2])
    max_layers = int(np.log2(min_dim)) if min_dim > 0 else 1
    num_conv_layers = min(requested_layers, max_layers)
    
    if num_conv_layers < requested_layers:
        print(
            f"[Builder Warning] 3D_CNN: Requested {requested_layers} layers, but data dimension only supports {num_conv_layers}. Adjusting automatically."
        )

    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Build convolutional layers with residual connections
    for i in range(num_conv_layers):
        # Filter configuration - controlled growth
        num_filters = min(512, params.get("filters", 32) * (2**(i // 2)))  # Grow slower
        
        # Residual shortcut
        shortcut = x
        
        # First conv block
        x = layers.Conv3D(
            num_filters, 
            kernel_size=params.get("kernel_size", 3), 
            padding="same", 
            kernel_regularizer=reg,
            bias_regularizer=reg
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Second conv block for residual module
        x = layers.Conv3D(
            num_filters, 
            kernel_size=params.get("kernel_size", 3), 
            padding="same", 
            kernel_regularizer=reg,
            bias_regularizer=reg
        )(x)
        x = layers.BatchNormalization()(x)
        
        # Match dimensions for shortcut if needed
        if shortcut.shape[-1] != num_filters:
            shortcut = layers.Conv3D(
                num_filters, 
                kernel_size=1, 
                padding="same",
                kernel_regularizer=reg
            )(shortcut)
            
        # Add residual connection
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        
        # Pooling - more conservative approach to preserve spatial info
        if i % 2 == 1 or i == num_conv_layers - 1:  # Pool every other layer
            x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
            
    # Global average pooling
    x = layers.GlobalAveragePooling3D()(x)
    
    # Dense layers with dropout
    dense_units = params.get("dense_units", 256)
    dropout_rate = params.get("dropout_rate", 0.4)  # Mid-range dropout
    
    x = layers.Dense(dense_units, kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    
    model = models.Model(inputs, outputs)
    
    # Optimizer with low learning rate and weight decay
    learning_rate = params.get("learning_rate", 1e-4)
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        weight_decay=params.get("weight_decay", 1e-5)
    )
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model