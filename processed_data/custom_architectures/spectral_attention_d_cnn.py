def build_model(input_shape, num_classes, params):
    """
    Advanced 1D CNN with Spectral Attention (SE-blocks), Dilated Convolutions,
    and Residual Connections. Optimized for spectral data (e.g., SERS).
    
    Design Strategy Implementation:
    - High Dropout (0.8) + Low Weight Decay (0.001) regularization
    - 6-layer depth with 48 filters as architectural anchor
    - Squeeze-and-Excitation (SE) blocks for spectral attention
    - Dilated convolutions for increased receptive field
    - Cosine annealing learning rate schedule
    - Dense layer constraint (>= 32 units enforced)
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np
    
    # --- Squeeze-and-Excitation Block for 1D (Spectral Attention) ---
    def se_block_1d(input_tensor, reduction_ratio=16):
        """Channel-wise attention mechanism for 1D convolutions."""
        filters = input_tensor.shape[-1]
        # Squeeze: Global Average Pooling
        se = layers.GlobalAveragePooling1D()(input_tensor)
        se = layers.Reshape((1, filters))(se)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        se = layers.Dense(
            filters // reduction_ratio,
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=False
        )(se)
        se = layers.Dense(
            filters,
            activation='sigmoid',
            kernel_initializer='he_normal',
            use_bias=False
        )(se)
        # Scale: Channel-wise multiplication
        return layers.Multiply()([input_tensor, se])
    
    # --- Parameter Extraction with Design Strategy Defaults ---
    max_layers = int(np.log2(input_shape[0])) if input_shape[0] > 0 else 1
    requested_layers = params.get("num_conv_layers", 6)
    num_conv_layers = min(requested_layers, max_layers)
    
    if num_conv_layers < requested_layers:
        print(f"[Builder Warning] 1D_CNN: Requested {requested_layers} layers, "
              f"but data dimension only supports {num_conv_layers}. Adjusting automatically.")
    
    # Architectural anchors from Design Strategy
    filters = params.get("filters", 48)
    kernel_size = params.get("kernel_size", 7)  # Configurable: 5, 7, or 9
    dense_units = max(32, params.get("dense_units", 48))  # Enforce constraint: never < 32
    
    # Regularization: High Dropout, Low Decay (dropout=0.8, weight_decay=0.001)
    dropout_rate = params.get("dropout_rate", 0.8)
    weight_decay = params.get("weight_decay", 0.001)
    reg = keras.regularizers.l2(weight_decay)
    
    # Learning parameters
    learning_rate = params.get("learning_rate", 0.0001)
    use_cosine_decay = params.get("use_cosine_decay", True)
    decay_steps = params.get("decay_steps", 1000)
    
    # Architecture options
    use_residual = params.get("use_residual", True)
    use_spectral_attention = params.get("use_spectral_attention", True)
    use_dilation = params.get("use_dilation", True)  # Enable dilated convs per strategy
    grow_filters = params.get("grow_filters", False)
    
    # --- Model Construction ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Data Augmentation (Gaussian Noise for spectral stability)
    if params.get("use_data_augmentation", False):
        x = layers.GaussianNoise(0.1)(x)
    
    # Initial projection to target filter dimension
    x = layers.Conv1D(
        filters, 
        kernel_size, 
        padding="same",
        kernel_regularizer=reg,
        kernel_initializer='he_normal'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Residual Convolutional Blocks (6 layers = 3 blocks × 2 layers)
    num_blocks = num_conv_layers // 2
    for i in range(num_blocks):
        # Calculate current filters (fixed 48 or growing)
        current_filters = min(512, filters * (2**i)) if grow_filters else filters
        
        # Dilation rate: Exponentially increase to expand receptive field without adding layers
        # Dilation sequence: 1, 2, 4 for 3 blocks (receptive field grows exponentially)
        dilation_rate = 2**i if use_dilation else 1
        
        # Residual shortcut
        shortcut = x
        
        # --- First Conv in Block ---
        x = layers.Conv1D(
            current_filters,
            kernel_size,
            padding="same",
            dilation_rate=dilation_rate,
            kernel_regularizer=reg,
            kernel_initializer='he_normal'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # --- Second Conv in Block ---
        x = layers.Conv1D(
            current_filters,
            kernel_size,
            padding="same",
            dilation_rate=dilation_rate,
            kernel_regularizer=reg,
            kernel_initializer='he_normal'
        )(x)
        x = layers.BatchNormalization()(x)
        
        # --- Residual Connection ---
        if use_residual:
            # Match dimensions if needed
            if shortcut.shape[-1] != current_filters:
                shortcut = layers.Conv1D(
                    current_filters, 
                    1, 
                    padding="same",
                    kernel_regularizer=reg,
                    kernel_initializer='he_normal'
                )(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            x = layers.Add()([x, shortcut])
        
        x = layers.Activation("relu")(x)
        
        # --- Spectral Attention (SE Block) ---
        if use_spectral_attention:
            x = se_block_1d(x, reduction_ratio=16)
        
        # --- Downsampling (except after last block) ---
        if i < num_blocks - 1:
            x = layers.MaxPooling1D(2)(x)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification Head with enforced dense layer
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=reg,
        kernel_initializer='he_normal'
    )(x)
    # High dropout (0.8) for aggressive regularization against overfitting
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    x = layers.Dense(num_classes, kernel_regularizer=reg)(x)
    outputs = layers.Activation("softmax", dtype="float32")(x)
    
    # --- Optimizer with Cosine Annealing ---
    if use_cosine_decay:
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            alpha=0.1  # Minimum LR = 0.1 * initial_learning_rate
        )
    else:
        lr_schedule = learning_rate
    
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model