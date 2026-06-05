def build_model(input_shape, num_classes, params):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # --- Architecture Constraints based on Pattern Analysis ---
    # Enforce maximum 3 conv layers (optimal: 2). Deep networks (>3) consistently overfit.
    max_layers = min(3, int(np.log2(input_shape[0]))) if input_shape[0] > 0 else 1
    requested_layers = params.get("num_conv_layers", 2)
    num_conv_layers = min(requested_layers, max_layers)
    
    if num_conv_layers < requested_layers:
        print(
            f"[Builder Warning] Enhanced 1D CNN: Requested {requested_layers} layers, "
            f"but constrained to {num_conv_layers} (max: 3) to prevent overfitting."
        )
    
    # --- Hyperparameters with Anti-Overfitting Constraints ---
    # Motif A & B: Dropout must be >0.80 (0.811-0.889 range). Enforce floor at 0.85.
    dropout_rate = max(0.85, min(0.9, params.get("dropout_rate", 0.85)))
    
    # Cap filters at 64 (Anti-Motif: >64 filters causes overfitting with history length 90)
    base_filters = min(64, params.get("filters", 32))
    
    # Restrict kernel size to {5, 7} (Anti-Motif: kernel 9 extracts spurious correlations)
    kernel_size = params.get("kernel_size", 5)
    if kernel_size not in [5, 7]:
        kernel_size = 5  # Default to safest option
        
    # Cap dense units at 64 (Anti-Motif: 128 units creates 1M+ parameter bottleneck)
    dense_units = min(64, params.get("dense_units", 32))
    
    # Advanced regularization parameters
    l2_lambda = params.get("l2_lambda", 1e-4)  # Range: 1e-4 to 1e-3
    use_spatial_dropout = params.get("use_spatial_dropout", True)
    use_residual = params.get("use_residual", True)  # Enable for gradient flow if 3 layers
    
    # Optimal learning rate from Motif A (~7e-05)
    learning_rate = params.get("learning_rate", 7e-05)
    
    # --- Model Construction ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Data Augmentation: Gaussian Noise for robustness
    if params.get("use_data_augmentation"):
        x = layers.GaussianNoise(0.1)(x)
    
    # L2 Regularizer for weight decay
    reg = keras.regularizers.l2(l2_lambda)
    
    # Convolutional Blocks with strict capacity controls
    for i in range(num_conv_layers):
        # Progression: 32 -> 64 filters max (never exceed 64 per Anti-Motif 2)
        current_filters = min(64, base_filters * (2**i))
        
        # Residual shortcut for gradient preservation (especially important if attempting 3 layers)
        shortcut = x
        
        # Conv -> BN -> Activation
        x = layers.Conv1D(
            filters=current_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=reg,
            kernel_initializer="he_normal"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Spatial Dropout: More appropriate than standard dropout for CNN feature maps
        # Drops entire 1D feature maps (channels) rather than individual elements
        if use_spatial_dropout:
            x = layers.SpatialDropout1D(dropout_rate)(x)
        
        # Residual connection with projection if dimensions mismatch
        if use_residual:
            if shortcut.shape[-1] != current_filters:
                shortcut = layers.Conv1D(
                    filters=current_filters,
                    kernel_size=1,
                    padding="same",
                    kernel_regularizer=reg,
                    kernel_initializer="he_normal"
                )(shortcut)
            x = layers.Add()([x, shortcut])
        
        # Spatial downsampling
        x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Global Average Pooling: Reduces parameters vs Flatten (parameter budget <100K)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classifier head with strict capacity limit (Motif A: 32 units, Motif B: 64 units)
    x = layers.Dense(
        units=dense_units,
        activation="relu",
        kernel_regularizer=reg,
        kernel_initializer="he_normal"
    )(x)
    # Standard dropout for 2D tensor (post-pooling)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile with Adam optimizer (tuned LR from successful motifs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model