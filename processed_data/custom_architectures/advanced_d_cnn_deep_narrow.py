def build_model(input_shape, num_classes, params: dict):
    """
    Advanced 1D CNN with Deep-Narrow Architecture and Residual Connections.
    
    Optimized for SERS spectral classification based on empirical pattern analysis:
    - Deep-Narrow: 6 convolutional layers with fixed 32 filters
    - Micro-receptive fields: Kernel size 3 exclusively
    - Residual connections every 2 layers to enable gradient flow
    - Aggressive regularization: Weight decay (L2) primary, modest dropout secondary
    - Ultra-low learning rate: 5e-5 for stable convergence
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np
    
    # Defensive cast for input shape
    input_shape = tuple(input_shape)
    
    # Architecture parameters based on Pattern Analysis
    num_blocks = 3  # 3 blocks × 2 conv layers = 6 total conv layers
    filters = 32    # Fixed narrow width (Deep-Narrow motif)
    kernel_size = 3 # Micro-receptive field
    weight_decay = params.get("weight_decay", 1e-4)  # L2 regularization
    dropout_rate = params.get("dropout_rate", 0.25)  # Modest dropout (0.2-0.3 range)
    dense_units = params.get("dense_units", 64)      # Moderate dense layer
    learning_rate = params.get("learning_rate", 5e-5) # Ultra-low LR
    use_residual = params.get("use_residual", True)   # Residual every 2 layers
    use_augmentation = params.get("use_data_augmentation", False)
    
    # Regularizer
    reg = keras.regularizers.l2(weight_decay)
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Data Augmentation (1D Gaussian Noise)
    if use_augmentation:
        x = layers.GaussianNoise(0.1)(x)
    
    # Deep-Narrow Architecture: 6 layers grouped in pairs with residual connections
    for block_idx in range(num_blocks):
        # Store input for residual connection (every 2 layers)
        shortcut = x
        
        # First conv in block
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=reg,
            use_bias=False  # BN handles bias
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Second conv in block
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=reg,
            use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Residual connection every 2 layers (end of block)
        if use_residual:
            # Project shortcut if dimensions don't match (first block or channel change)
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv1D(
                    filters=filters,
                    kernel_size=1,
                    padding="same",
                    kernel_regularizer=reg
                )(shortcut)
            
            # Add residual connection before final activation of block
            x = layers.Add()([x, shortcut])
        
        # MaxPooling after every 2 conv layers (maintains progressive downsampling)
        x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Global Average Pooling (reduces parameters vs Flatten)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification Head: Moderate dense layer (64 units)
    x = layers.Dense(
        units=dense_units,
        activation="relu",
        kernel_regularizer=reg
    )(x)
    
    # Modest dropout for regularization (complements weight decay)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation("softmax", dtype="float32")(x)
    
    # Model compilation with ultra-low learning rate
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        # No warmup needed based on convergence stability metrics
    )
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model