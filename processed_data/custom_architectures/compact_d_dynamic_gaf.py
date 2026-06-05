def build_model(input_shape, num_classes, params: dict):
    """
    Compact 3D Dynamic GAF Architecture optimized for the "Small & Steady" paradigm.
    
    Design based on motif analysis showing optimal performance with:
    - 1.2M-5.4M parameters (constant 32 filters, 64 dense units, 6 conv layers)
    - Extended training horizon support (300+ epochs, lr=1e-5)
    - Dropout 0.4 sweet spot (no L2 regularization)
    - Spatial-only pooling (1,2,2) to preserve 304-length temporal history
    - Flatten instead of GlobalAveragePooling to maintain parameter count in optimal range
    
    Args:
        input_shape: Tuple of (time_steps, height, width, channels)
        num_classes: Number of output classes
        params: Dict containing hyperparameters (filters, kernel_size, dense_units, 
                dropout_rate, num_conv_layers, use_residual, learning_rate)
    
    Returns:
        Compiled Keras Model
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras

    # Architectural parameters from successful motif analysis
    filters = params.get("filters", 32)  # Fixed 32 filters - avoid width expansion
    kernel_size = params.get("kernel_size", 3)  # 3 or 5 both effective in compact regime
    dense_units = params.get("dense_units", 64)  # 64 units optimal (not 128+)
    dropout_rate = params.get("dropout_rate", 0.4)  # 0.4 sweet spot (not 0.5+)
    num_conv_layers = params.get("num_conv_layers", 6)  # 5-7 layers depth optimal
    use_residual = params.get("use_residual", False)
    learning_rate = params.get("learning_rate", 1e-5)  # Ultra-low LR for GAF convergence
    
    # Defensive: Calculate max layers based on spatial dimensions
    # Using pool_size=(1,2,2) preserves temporal, pools spatial by 2x per layer
    min_spatial_dim = min(input_shape[1], input_shape[2])
    max_layers = int(np.log2(min_spatial_dim)) if min_spatial_dim > 0 else 1
    requested_layers = num_conv_layers
    
    if requested_layers > max_layers:
        print(f"[Builder Warning] 3D_DYNAMIC_GAF: Requested {requested_layers} layers, "
              f"but spatial dim {min_spatial_dim} only supports {max_layers}. "
              f"Adjusting automatically to prevent spatial collapse.")
        num_conv_layers = max_layers
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Build deep compact feature extractor (depth over width)
    for i in range(num_conv_layers):
        shortcut = x
        
        # Constant 32 filters (no doubling) to stay in 1M-6M param range
        x = keras.layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            use_bias=False,  # BatchNorm handles bias
            kernel_regularizer=None  # Explicitly no L2 per motif analysis
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Residual connection (optional) - simplified since channels constant
        if use_residual:
            if shortcut.shape[-1] != filters:
                shortcut = keras.layers.Conv3D(
                    filters, kernel_size=1, padding="same", use_bias=False
                )(shortcut)
            x = keras.layers.Add()([x, shortcut])
        
        # Pool spatial dimensions only (1, 2, 2) to preserve temporal structure
        # Critical for 304-length history mentioned in strategy
        x = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # Use Flatten instead of GlobalAveragePooling3D to maintain 1.2M+ parameters
    # This preserves temporal-spatial structure for the dense layer
    x = keras.layers.Flatten()(x)
    
    # Compact bottleneck features (64 units) - implicit regularization
    x = keras.layers.Dense(
        dense_units, 
        activation="relu",
        kernel_regularizer=None  # No weight decay per successful trials
    )(x)
    
    # Dropout 0.4: Optimal bias-variance tradeoff for GAF representation
    x = keras.layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Optimizer: Adam with ultra-low learning rate for extended training (300+ epochs)
    # Supports cosine annealing from 1e-5 to 1e-6 if implemented in training loop
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Log parameter count for verification (target: 1M-6M)
    param_count = model.count_params()
    if param_count < 1_000_000 or param_count > 6_000_000:
        print(f"[Builder Warning] Parameter count {param_count:,} outside optimal "
              f"1M-6M range. Current config: {num_conv_layers} layers, {filters} filters, "
              f"{dense_units} dense units.")
    else:
        print(f"[Builder Info] Parameter count {param_count:,} in optimal 1M-6M range.")
    
    return model