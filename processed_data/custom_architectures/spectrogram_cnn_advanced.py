def build_model(input_shape, num_classes, params):
    """
    Timeout-Optimized 2D Spectrogram CNN for 2D_SPECTROGRAM representation.
    Reduced complexity to prevent training timeouts while maintaining performance.
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Timeout-Optimized Configuration (Reduced from original to prevent timeouts)
    kernel_size = 3
    
    # Reduce default layers for faster training (4 instead of 6)
    min_dim = min(input_shape[0], input_shape[1])
    max_layers = int(np.log2(min_dim)) if min_dim > 0 else 1
    requested_layers = params.get("num_conv_layers", 4)  # Reduced default
    num_conv_layers = min(requested_layers, max_layers, 4)  # Hard cap at 4 for speed
    
    if num_conv_layers < requested_layers:
        print(f"[Architect Warning] Reduced conv layers from {requested_layers} to {num_conv_layers} to prevent timeout.")
    
    # Reduce default filters for faster training (64 instead of 128)
    base_filters = params.get("filters", 64)  # Reduced default
    max_filters = 64  # Lower cap for speed (was 128)
    num_filters = min(base_filters, max_filters)
    
    # Reduce dense units for faster training (256 instead of 512)
    dense_units = min(params.get("dense_units", 256), 256)  # Reduced cap
    
    # Conservative hyperparameters
    dropout_rate = params.get("dropout_rate", 0.3)  # Slightly reduced for faster convergence
    learning_rate = params.get("learning_rate", 1e-4)  # Slightly higher for faster learning (was 1e-5)
    use_residual = params.get("use_residual", True)
    use_se = params.get("use_squeeze_excite", False)  # Disabled by default to save computation time
    use_augmentation = params.get("use_data_augmentation", False)  # Disabled to prevent timeout
    l2_factor = params.get("l2_reg", 1e-5)  # Reduced regularization
    
    # --- Build Model ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Data Augmentation (Disabled by default to prevent timeout)
    if use_augmentation:
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.05)(x)
        x = layers.RandomZoom(0.1)(x)
    
    # Regularization
    reg = keras.regularizers.l2(l2_factor) if l2_factor > 0 else None
    
    # Convolutional Backbone (Optimized for speed)
    current_filters = num_filters
    
    for i in range(num_conv_layers):
        shortcut = x
        
        # Main conv path
        x = layers.Conv2D(
            current_filters,
            kernel_size,
            padding="same",
            kernel_regularizer=reg,
            use_bias=False
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Squeeze-and-Excitation (Optional, disabled by default for speed)
        if use_se:
            se_channels = max(1, current_filters // 16)
            se = layers.GlobalAveragePooling2D()(x)
            se = layers.Reshape((1, 1, current_filters))(se)
            se = layers.Conv2D(se_channels, 1, activation="relu", kernel_regularizer=reg)(se)
            se = layers.Conv2D(current_filters, 1, activation="sigmoid", kernel_regularizer=reg)(se)
            x = layers.multiply([x, se])
        
        # Residual connection
        if use_residual:
            if shortcut.shape[-1] != current_filters:
                shortcut = layers.Conv2D(
                    current_filters, 
                    (1, 1), 
                    padding="same", 
                    kernel_regularizer=reg,
                    use_bias=False
                )(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            x = layers.Add()([x, shortcut])
        
        # Spatial downsampling (every layer to reduce spatial dims quickly)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Classifier Head (Lightweight)
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=reg
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Optimizer: Adam with slightly higher LR for faster convergence
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Log architecture stats
    param_count = model.count_params()
    print(f"[Architect Info] Model configured with {param_count:,} parameters.")
    if param_count > 5_000_000:
        print(f"[Architect Warning] Model has {param_count:,} params (>5M). Risk of timeout.")
    
    return model