def build_model(input_shape, num_classes, params):
    """
    Advanced 2D GAF CNN Architecture.
    
    Implements deep 7-layer convolutional stacks with large receptive fields (kernel=7),
    aggressive regularization (L2=0.01, Dropout=0.65), Squeeze-and-Excitation blocks,
    and progressive filter doubling (32->64->128...->512) for hierarchical feature 
    extraction from Gramian Angular Field representations.
    
    Key improvements over base 2D_GAF:
    - 7 convolutional layers with kernel size 7 for large receptive fields
    - Progressive filter expansion (32 to 512) capped at capacity sweet spot
    - Squeeze-and-Excitation attention blocks for channel recalibration
    - Aggressive regularization stack: L2(0.01) + Dropout(0.65) + SpatialDropout2D(0.15)
    - AdamW optimizer with weight decay 0.01 for improved generalization
    - Batch Normalization with momentum 0.9 for training stability
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # --- Architecture Parameters (from Design Strategy) ---
    num_conv_layers = params.get("num_conv_layers", 7)  # Deep stack: 7 layers optimal
    kernel_size = params.get("kernel_size", 7)          # Large receptive field: 7x7
    base_filters = params.get("filters", 32)            # Start of progression: 32->64->128...
    dense_units = params.get("dense_units", 512)        # Capacity sweet spot: 512
    dropout_rate = params.get("dropout_rate", 0.65)     # Aggressive dropout: 0.6-0.7 range
    l2_reg_strength = params.get("l2_reg", 0.01)        # High L2: 0.01 for generalization
    spatial_dropout_rate = params.get("spatial_dropout_rate", 0.15)  # 0.1-0.2 for conv layers
    use_se = params.get("use_squeeze_excitation", True) # Enable SE blocks
    use_progressive = params.get("use_progressive_filters", True)
    use_residual = params.get("use_residual", True)     # Residual connections for 7-layer depth
    
    # Defensive calculation for max layers (allow 7 for 118x118 with 'same' pooling)
    min_dim = min(input_shape[0], input_shape[1])
    max_layers = int(np.log2(min_dim)) + 2 if min_dim > 0 else 1  # +2 allows 7 layers with 'same' padding
    
    if num_conv_layers > max_layers:
        print(f"[Advanced 2D GAF] Warning: Reduced layers from {num_conv_layers} to {max_layers} due to input size.")
        num_conv_layers = max_layers
    
    reg = keras.regularizers.l2(l2_reg_strength)
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # --- Data Augmentation (Conservative for GAF) ---
    # Horizontal flip corresponds to time reversal (valid for time series)
    # Avoid rotation/zoom which distort angular temporal relationships
    if params.get("use_data_augmentation", False):
        x = layers.RandomFlip("horizontal")(x)
    
    # --- Convolutional Stack with SE Blocks ---
    for i in range(num_conv_layers):
        # Progressive filter doubling: 32, 64, 128, 256, 512, 512, 512
        if use_progressive:
            num_filters = min(512, base_filters * (2 ** i))
        else:
            num_filters = params.get("static_filters", 64)
        
        shortcut = x
        
        # Convolutional block: Conv -> BN -> ReLU
        x = layers.Conv2D(
            num_filters,
            (kernel_size, kernel_size),
            strides=1,
            padding="same",
            kernel_regularizer=reg,
            kernel_initializer="he_normal",
            name=f"conv_{i+1}"
        )(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = layers.Activation("relu")(x)
        
        # Squeeze-and-Excitation block for channel attention
        if use_se:
            se = layers.GlobalAveragePooling2D()(x)
            se = layers.Dense(num_filters // 16, activation="relu", kernel_regularizer=reg)(se)  # Reduction ratio 16
            se = layers.Dense(num_filters, activation="sigmoid", kernel_regularizer=reg)(se)
            se = layers.Reshape((1, 1, num_filters))(se)
            x = layers.Multiply()([x, se])
            x = layers.BatchNormalization()(x)  # Stabilize after SE scaling
        
        # Spatial dropout for feature map regularization (light, 0.1-0.2)
        if spatial_dropout_rate > 0:
            x = layers.SpatialDropout2D(spatial_dropout_rate)(x)
        
        # Residual connection (essential for 7-layer depth)
        if use_residual:
            if shortcut.shape[-1] != num_filters:
                shortcut = layers.Conv2D(
                    num_filters, 
                    (1, 1), 
                    padding="same", 
                    kernel_regularizer=reg,
                    kernel_initializer="he_normal"
                )(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            x = layers.Add()([x, shortcut])
        
        # Pooling with 'same' padding to preserve depth capacity (7 layers)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    
    # --- Classifier Head ---
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=reg,
        kernel_initializer="he_normal",
        name="dense_512"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)  # Aggressive dropout: 0.65
    
    outputs = layers.Dense(
        num_classes, 
        activation="softmax",
        kernel_regularizer=reg,
        kernel_initializer="glorot_uniform"
    )(x)
    
    model = keras.Model(inputs, outputs)
    
    # --- Optimizer: AdamW with weight decay (Advanced technique) ---
    # Replaces manual L2 regularization with decoupled weight decay
    learning_rate = params.get("learning_rate", 1e-5)  # Optimal: 1e-5 for this depth
    optimizer_type = params.get("optimizer", "AdamW")
    
    if optimizer_type == "AdamW":
        optimizer = keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,  # As per strategy to replace L2
            clipnorm=1.0        # Gradient clipping for stability with aggressive regularization
        )
    else:
        # Fallback to standard Adam with explicit L2 (from reg layers)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model