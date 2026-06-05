def build_model(input_shape, num_classes, params: dict):
    """
    Optimized 2D Spectrogram CNN with 6-layer depth stability.
    
    Architecture based on Representation Stability findings:
    - 6 Convolutional layers (64 filters, 3x3) with BatchNorm
    - Global Average Pooling → Dense(256) → Dropout(0.6)
    - Optimized for ~1.6M param regime: LR 0.0005, Weight Decay 0.005
    - Batch size constraint: ≤32 (optimal 32 for this capacity)
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    
    # Defensive cast for input shape
    input_shape = tuple(input_shape)
    
    # --- Architecture Configuration ---
    # Enforce 6 layers as optimal depth (diminishing returns beyond 6)
    min_dim = min(input_shape[0], input_shape[1])
    max_layers = int(np.log2(min_dim)) if min_dim > 0 else 1
    requested_layers = params.get("num_conv_layers", 6)
    num_conv_layers = min(requested_layers, max_layers)
    
    if num_conv_layers < requested_layers:
        print(f"[Spectrogram CNN 6L] Warning: Reduced layers from {requested_layers} to {num_conv_layers} due to input size constraints.")
    
    # Fixed 64 filters per layer (constant, not doubling) per stability motif
    num_filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 3)
    
    # --- Regularization Configuration (1.6M param regime) ---
    # Moderate dropout (0.60), weight decay (0.005) to prevent underfitting
    dropout_rate = params.get("dropout_rate", 0.6)
    weight_decay = params.get("weight_decay", 0.005)
    reg = keras.regularizers.l2(weight_decay) if weight_decay > 0 else None
    
    # --- Model Construction ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Optional Data Augmentation (2D)
    if params.get("use_data_augmentation"):
        x = keras.layers.RandomFlip("horizontal")(x)
        x = keras.layers.RandomRotation(0.1)(x)
        x = keras.layers.RandomZoom(0.1)(x)
    
    # Six Conv Blocks: Conv2D → BatchNorm → ReLU → MaxPool
    # BatchNorm is non-negotiable for depth ≥6 to prevent instability
    for i in range(num_conv_layers):
        x = keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(kernel_size, kernel_size),
            padding="same",
            kernel_regularizer=reg,
            use_bias=False  # Bias false when using BatchNorm
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Global Pooling and Dense Head
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense(256) as specified in stability protocol
    dense_units = params.get("dense_units", 256)
    x = keras.layers.Dense(
        units=dense_units,
        activation="relu",
        kernel_regularizer=reg
    )(x)
    
    # Dropout(0.6) for moderate regularization in high-capacity regime
    x = keras.layers.Dropout(dropout_rate)(x)
    
    # Output layer with softmax
    x = keras.layers.Dense(num_classes)(x)
    outputs = keras.layers.Activation("softmax", dtype="float32")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Optimizer: Adam with LR 0.0005 for stable convergence
    learning_rate = params.get("learning_rate", 0.0005)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model