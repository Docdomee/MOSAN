def build_model(input_shape, num_classes, params):
    """
    Advanced 2D Spectrogram CNN implementing the "Stable Six" Goldilocks configuration
    with optional Squeeze-and-Excitation blocks and residual connections.
    
    Architecture defaults based on Pattern Analysis:
    - 6 convolutional layers (depth optimum)
    - 128 filters per layer (constant width, avoid doubling)
    - 3x3 kernels exclusively (receptive field optimum)
    - 512 dense units (balanced capacity)
    - 0.5 dropout (optimal regularization)
    - 1e-5 learning rate (conservative optimization)
    
    Advanced techniques (via params):
    - use_squeeze_excite: Enable SE blocks for channel attention
    - use_residual: Enable skip connections (required for depth >6)
    - use_progressive_dropout: Linearly increase dropout 0.3->0.5 across conv layers
    
    Representation: 2D_SPECTROGRAM
    """
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    
    # --- Stable Six Core Configuration ---
    num_conv_layers = int(params.get("num_conv_layers", 6))
    num_filters = int(params.get("filters", 128))
    kernel_size = int(params.get("kernel_size", 3))
    dense_units = int(params.get("dense_units", 512))
    dropout_rate = float(params.get("dropout_rate", 0.5))
    learning_rate = float(params.get("learning_rate", 1e-5))
    
    # --- Advanced Technique Flags ---
    use_residual = bool(params.get("use_residual", False))
    use_se = bool(params.get("use_squeeze_excite", False))
    se_ratio = int(params.get("se_ratio", 16))
    use_progressive_dropout = bool(params.get("use_progressive_dropout", False))
    
    # --- Defensive Architecture Constraints ---
    # Handle None dimensions in input_shape for 2D_SPECTROGRAM (height, width, channels)
    if len(input_shape) < 2:
        raise ValueError(f"2D_SPECTROGRAM requires at least 2D input shape, got {input_shape}")
    
    h, w = input_shape[0], input_shape[1]
    
    if h is None or w is None:
        max_possible_layers = num_conv_layers
    else:
        min_dim = min(int(h), int(w))
        max_possible_layers = int(np.log2(max(min_dim, 2)))
        
        if num_conv_layers > max_possible_layers:
            print(f"[Architect Warning] Input {input_shape} supports max {max_possible_layers} layers. "
                  f"Reducing {num_conv_layers} -> {max_possible_layers}.")
            num_conv_layers = max(1, max_possible_layers)
    
    # Pattern Scout: Do not exceed 6 layers without residual connections
    if num_conv_layers > 6 and not use_residual:
        print(f"[Architect Warning] Depth {num_conv_layers} >6 requires residuals. Enabling residuals.")
        use_residual = True
    
    # --- Build Model ---
    inputs = keras.Input(shape=input_shape, name="spectrogram_input")
    x = inputs
    
    # Data Augmentation (2D) - Only if explicitly requested
    if params.get("use_data_augmentation", False):
        x = keras.layers.RandomFlip("horizontal", name="rand_flip")(x)
        x = keras.layers.RandomRotation(0.1, name="rand_rot")(x)
        x = keras.layers.RandomZoom(0.1, name="rand_zoom")(x)
    
    reg = keras.regularizers.l2(float(params.get("l2_reg", 1e-4)))
    
    # --- Convolutional Backbone: "6-128-3" ---
    for i in range(num_conv_layers):
        shortcut = x
        block_name = f"conv_block_{i+1}"
        
        # Progressive dropout calculation
        if use_progressive_dropout:
            current_dropout = 0.3 + (0.2 * i / max(num_conv_layers - 1, 1))
        else:
            current_dropout = 0.0
        
        # Main convolutional path
        x = keras.layers.Conv2D(
            num_filters,
            (kernel_size, kernel_size),
            padding="same",
            kernel_regularizer=reg,
            use_bias=False,
            name=f"{block_name}_conv"
        )(x)
        x = keras.layers.BatchNormalization(name=f"{block_name}_bn")(x)
        x = keras.layers.Activation("relu", name=f"{block_name}_relu")(x)
        
        # Squeeze-and-Excitation Block (Channel Attention)
        if use_se:
            se = keras.layers.GlobalAveragePooling2D(name=f"{block_name}_se_gap")(x)
            se = keras.layers.Reshape((1, 1, num_filters), name=f"{block_name}_se_reshape")(se)
            se = keras.layers.Dense(
                max(1, num_filters // se_ratio),
                activation="relu",
                kernel_regularizer=reg,
                name=f"{block_name}_se_reduce"
            )(se)
            se = keras.layers.Dense(
                num_filters,
                activation="sigmoid",
                kernel_regularizer=reg,
                name=f"{block_name}_se_expand"
            )(se)
            x = keras.layers.Multiply(name=f"{block_name}_se_scale")([x, se])
        
        # Spatial dropout (progressive)
        if use_progressive_dropout and current_dropout > 0:
            x = keras.layers.SpatialDropout2D(current_dropout, name=f"{block_name}_sdrop")(x)
        
        # Residual Connection with dimension matching
        if use_residual:
            # Match channel dimensions with 1x1 projection if necessary
            # Safely handle None channel dimensions during model building
            shortcut_channels = shortcut.shape[-1]
            if shortcut_channels is None or shortcut_channels != num_filters:
                shortcut = keras.layers.Conv2D(
                    num_filters,
                    (1, 1),
                    padding="same",
                    kernel_regularizer=reg,
                    use_bias=False,
                    name=f"{block_name}_proj"
                )(shortcut)
                shortcut = keras.layers.BatchNormalization(name=f"{block_name}_proj_bn")(shortcut)
            
            # Add residual only if spatial dimensions match (pre-pooling)
            # Use shape comparison compatible with None dimensions
            shortcut_shape = shortcut.shape[1:3]
            x_shape = x.shape[1:3]
            
            # Check compatibility (allow None to match None or specific value)
            shapes_compatible = (
                (shortcut_shape[0] is None or x_shape[0] is None or shortcut_shape[0] == x_shape[0]) and
                (shortcut_shape[1] is None or x_shape[1] is None or shortcut_shape[1] == x_shape[1])
            )
            
            if shapes_compatible:
                x = keras.layers.Add(name=f"{block_name}_residual")([x, shortcut])
        
        # Pooling (2x2 reduction)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), name=f"{block_name}_pool")(x)
    
    # --- Classifier Head ---
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    x = keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_1"
    )(x)
    
    x = keras.layers.Dropout(dropout_rate, name="dropout")(x)
    
    outputs = keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=reg,
        name="predictions"
    )(x)
    
    model = keras.Model(inputs, outputs, name="Stable_Six_2D_Spectrogram_CNN")
    
    # --- Conservative Optimization ---
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model