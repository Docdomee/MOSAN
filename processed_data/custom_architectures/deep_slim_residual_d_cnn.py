def build_model(input_shape, num_classes, params: dict):
    """
    Deep-Slim-Regularized 1D-CNN with Spectral Attention and Residual Connections.
    
    Architecture based on the "Deep-Slim-Regularized" motif optimized for SERS spectral data:
    - 6 convolutional layers with constant 48 filters (parameter efficient)
    - Kernel size 7 optimized for 70-length spectral windows
    - Residual connections every 2 layers to enable deep gradient flow
    - Spectral attention (SE-blocks) after layers 3 and 6 for feature recalibration
    - Aggressive regularization: Dropout 0.75 + Weight Decay 0.0025
    - Slim classifier: 48 dense units (reduced from 64 to prevent overfitting)
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # --- Optimal Hyperparameters from Pattern Scout Analysis ---
    num_conv_layers = params.get("num_conv_layers", 6)
    num_filters = params.get("filters", 48)  # Constant 48 filters (not growing)
    kernel_size = params.get("kernel_size", 7)  # Locked optimal size for spectral data
    dense_units = params.get("dense_units", 48)  # Slim: 48 units (not 64)
    dropout_rate = params.get("dropout_rate", 0.75)  # Sweet spot for regularization
    weight_decay = params.get("weight_decay", 0.0025)  # Must pair with high dropout
    learning_rate = params.get("learning_rate", 0.00013)  # Stable convergence
    use_residual = params.get("use_residual", True)  # Essential for 6+ layers
    use_spectral_attention = params.get("use_spectral_attention", True)  # Innovation
    use_data_augmentation = params.get("use_data_augmentation", False)
    
    # L2 Regularizer (Weight Decay)
    reg = keras.regularizers.l2(weight_decay)
    
    # Defensive: Limit layers based on input dimension to prevent collapse
    max_possible_layers = int(np.log2(input_shape[0])) if input_shape[0] > 0 else 1
    actual_layers = min(num_conv_layers, max_possible_layers)
    
    if actual_layers < num_conv_layers:
        print(f"[Deep-Slim-1D-CNN] Warning: Reduced layers from {num_conv_layers} to {actual_layers} due to input size.")
    
    # --- Input Block ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Optional Data Augmentation (Gaussian Noise for spectral robustness)
    if use_data_augmentation:
        x = layers.GaussianNoise(stddev=0.1)(x)
    
    # --- Deep Convolutional Stack ---
    for i in range(actual_layers):
        shortcut = x
        
        # Convolution: Fixed 48 filters, kernel 7, 'same' padding
        x = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=reg,
            kernel_initializer="he_normal"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Residual Connection (Pre-activation style with projection if needed)
        if use_residual:
            # Project shortcut if channel dimension mismatch
            if shortcut.shape[-1] != num_filters:
                shortcut = layers.Conv1D(
                    num_filters,
                    kernel_size=1,
                    padding="same",
                    kernel_regularizer=reg
                )(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            # Add residual before downsampling
            x = layers.Add()([x, shortcut])
        
        # Downsampling (MaxPool) - reduces sequence length by half
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        # Spectral Attention Mechanism (SE-block adapted for 1D)
        # Applied after layer 3 (index 2) and final layer (index 5) as per strategy
        if use_spectral_attention and (i == 2 or i == actual_layers - 1):
            # Squeeze: Global Average Pooling across spectral dimension
            se = layers.GlobalAveragePooling1D()(x)
            
            # Excitation: Bottleneck with reduction ratio 8 (48//8 = 6)
            se = layers.Dense(num_filters // 8, activation="relu")(se)
            se = layers.Dense(num_filters, activation="sigmoid")(se)
            se = layers.Reshape((1, num_filters))(se)
            
            # Scale: Channel-wise multiplication
            x = layers.Multiply()([x, se])
    
    # --- Slim Classification Head ---
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layer with aggressive regularization
    x = layers.Dense(
        units=dense_units,
        activation="relu",
        kernel_regularizer=reg,
        kernel_initializer="he_normal"
    )(x)
    
    # High dropout (0.75) as per regularization cocktail requirements
    x = layers.Dropout(rate=dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        dtype="float32",
        kernel_regularizer=reg
    )(x)
    
    # --- Model Compilation ---
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model