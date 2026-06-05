def build_model(input_shape, num_classes, params: dict):
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    
    # --- Design Strategy Parameters (per Critical Pattern Analysis) ---
    # Block-based architecture: 2 blocks with 32 init_channels is the sweet spot (~15K params)
    num_blocks = params.get("num_blocks", 2)
    init_channels = params.get("init_channels", 32)
    
    # Aggressive Regularization: 0.8 dropout and 0.01 weight decay essential for generalization
    dropout_rate = params.get("dropout_rate", 0.8)
    weight_decay = params.get("weight_decay", 0.01)
    
    # Fixed Low Learning Rate: 0.0001 for stable convergence (avoid 0.0005 which overfits)
    learning_rate = params.get("learning_rate", 0.0001)
    
    # Other params
    use_data_augmentation = params.get("use_data_augmentation", False)
    use_residual = params.get("use_residual", True)
    kernel_size = params.get("kernel_size", 3)
    
    # Safety check: Ensure we don't pool below 1 time step (history-16 supports max 4 blocks)
    max_blocks = int(np.log2(input_shape[0])) if input_shape[0] > 0 else 1
    if num_blocks > max_blocks:
        print(f"[Builder Warning] CompactBlockCNN: Requested {num_blocks} blocks, but input length {input_shape[0]} only supports {max_blocks}. Adjusting.")
        num_blocks = max_blocks
    
    # L2 Regularizer (Weight Decay) - Essential per strategy
    reg = keras.regularizers.l2(weight_decay)
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Optional Data Augmentation (Gaussian Noise safe for 1D signals)
    if use_data_augmentation:
        x = keras.layers.GaussianNoise(0.1)(x)
    
    # --- Compact Block Architecture ---
    # Strategy: Shallow (2 blocks) and wide-through-blocks rather than deep
    for i in range(num_blocks):
        # Double channels each block: 32 -> 64 (keeps params low, avoids >600K over-parameterization)
        filters = min(512, init_channels * (2**i))
        
        shortcut = x
        
        # Main Conv Path
        x = keras.layers.Conv1D(
            filters,
            kernel_size,
            padding="same",
            kernel_regularizer=reg,
            use_bias=False  # BatchNorm handles bias
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Residual Connection (optional but helps gradient flow in compact architectures)
        if use_residual:
            if shortcut.shape[-1] != filters:
                shortcut = keras.layers.Conv1D(
                    filters, 1, padding="same", kernel_regularizer=reg
                )(shortcut)
            x = keras.layers.Add()([x, shortcut])
        
        # Downsample and apply aggressive dropout between blocks
        x = keras.layers.MaxPooling1D(2)(x)
        if dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate)(x)
    
    # --- Classification Head ---
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # Aggressive regularization continued in dense layers
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(
        64,  # Strategy: Keep dense units <=64 (avoid large dense layers)
        activation="relu",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    
    # Output layer
    x = keras.layers.Dense(num_classes, kernel_regularizer=reg)(x)
    outputs = keras.layers.Activation("softmax", dtype="float32")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Optimizer with fixed low learning rate for stability
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model