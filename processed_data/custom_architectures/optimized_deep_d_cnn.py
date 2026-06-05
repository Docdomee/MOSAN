def build_model(input_shape, num_classes, params: dict):
    """
    Optimized 1D CNN for SERS Spectral Data.
    
    Architecture based on pattern analysis indicating:
    - Depth over width: 6 convolutional layers
    - Constant width: 48 filters per layer (hierarchical feature extraction)
    - Kernel size: 7 (optimal for 76-length spectral window)
    - Residual connections for gradient flow in deep configuration
    - Aggressive regularization: Dropout 0.72, Weight Decay 0.003
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # --- Architecture Parameters (Pattern Analysis Optimized) ---
    # Defensive layer calculation
    max_layers = int(np.log2(input_shape[0])) if input_shape[0] > 0 else 1
    requested_layers = params.get("num_conv_layers", 6)  # 6 layers optimal
    num_conv_layers = min(requested_layers, max_layers)
    
    if num_conv_layers < requested_layers:
        print(f"[Builder Warning] 1D_CNN: Requested {requested_layers} layers, but data dimension only supports {num_conv_layers}. Adjusting automatically.")
    
    # Constant 48 filters per layer (depth over width strategy)
    base_filters = params.get("filters", 48)
    kernel_size = params.get("kernel_size", 7)  # Kernel 7 for 76-length spectra
    
    # Regularization parameters (sweet spot from analysis)
    weight_decay = params.get("weight_decay", 0.003)  # 0.0025-0.004 range
    dropout_rate = params.get("dropout_rate", 0.72)   # 0.70-0.75 range
    dense_units = params.get("dense_units", 48)       # 48 units optimal
    learning_rate = params.get("learning_rate", 0.00013)  # ~0.00013 optimal
    
    # --- Model Construction ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Data Augmentation (optional)
    if params.get("use_data_augmentation", False):
        x = layers.GaussianNoise(0.1)(x)
    
    # Regularizer
    reg = keras.regularizers.l2(weight_decay) if weight_decay > 0 else None
    
    # Convolutional Stack: 6 layers × 48 filters
    for i in range(num_conv_layers):
        # Residual shortcut
        shortcut = x
        
        # Main convolutional block
        x = layers.Conv1D(
            filters=base_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=reg,
            kernel_initializer='he_normal'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Residual connection (enabled by default for deep networks)
        if params.get("use_residual", True):
            # Project shortcut if dimensions mismatch (first layer or channel changes)
            if shortcut.shape[-1] != base_filters:
                shortcut = layers.Conv1D(
                    filters=base_filters,
                    kernel_size=1,
                    padding="same",
                    kernel_initializer='he_normal',
                    kernel_regularizer=reg
                )(shortcut)
            x = layers.Add()([x, shortcut])
        
        # Downsampling
        x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Global feature aggregation
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head with controlled capacity
    x = layers.Dense(
        units=dense_units,
        activation="relu",
        kernel_regularizer=reg,
        kernel_initializer='he_normal'
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation("softmax", dtype="float32")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Optimizer with stable learning rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model