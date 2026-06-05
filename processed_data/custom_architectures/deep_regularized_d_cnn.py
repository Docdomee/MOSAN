def build_model(input_shape, num_classes, params: dict):
    """
    Deep-Regularized 1D CNN optimized for SERS spectral data.
    
    Architecture based on Pattern Analysis findings:
    - 4 Convolutional layers with fixed 48 filters (width)
    - Kernel size fixed at 7 (optimal receptive field)
    - Aggressive Dropout regime (0.55-0.60) with dual-dropout classifier stack
    - No BatchNorm, No Weight Decay (L2=0.0)
    - 128-unit Dense classifier head
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # --- Parameters from Design Strategy ---
    # Fixed architectural constants from winning trial (Trial 6)
    num_conv_layers = params.get("num_conv_layers", 4)
    filters = params.get("filters", 48)  # Fixed 48 filters per layer (not pyramid)
    kernel_size = params.get("kernel_size", 7)  # Fixed at 7 (5 insufficient, 9 reduces capacity)
    dense_units = params.get("dense_units", 128)  # 128 is the sweet spot (160 overfits)
    dropout_rate = params.get("dropout_rate", 0.55)  # Strictly >0.55 (0.5 fails catastrophically)
    learning_rate = params.get("learning_rate", 0.002)  # ~0.002 for fast stable convergence
    use_residual = params.get("use_residual", False)  # Disabled by default (avoid complexity)
    use_data_aug = params.get("use_data_augmentation", False)
    
    # Enforce depth constraint from input dimension (defensive)
    max_layers = int(np.log2(input_shape[0])) if input_shape[0] > 0 else 1
    if num_conv_layers > max_layers:
        print(f"[Builder Warning] 1D_CNN: Requested {num_conv_layers} layers, but data dimension only supports {max_layers}. Adjusting to {max_layers}.")
        num_conv_layers = max_layers
    
    # --- Input ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # --- Optional Data Augmentation ---
    if use_data_aug:
        x = layers.GaussianNoise(0.1)(x)
    
    # --- Convolutional Stack (4 layers, 48 filters, kernel 7) ---
    for i in range(num_conv_layers):
        shortcut = x
        
        # Fixed 48 filters as per optimal configuration (not increasing pyramid)
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=None,  # Explicitly disable: weight_decay=0.0 per pattern
            use_bias=True
        )(x)
        
        # BatchNorm REMOVED per pattern: "No BatchNorm (not tested, avoid complexity)"
        x = layers.Activation("relu")(x)
        
        # Optional residual (disabled by default to match strict pattern constraints)
        if use_residual:
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv1D(filters, 1, padding="same")(shortcut)
            x = layers.Add()([x, shortcut])
        
        # MaxPooling reduces sequence length by half each layer
        x = layers.MaxPooling1D(pool_size=2)(x)
    
    # --- Global Feature Aggregation ---
    x = layers.GlobalAveragePooling1D()(x)
    
    # --- Regularization Stack: Dropout 0.55 → Dense 128 → Dropout 0.55 ---
    # First dropout on pooled features (aggressive regularization per pattern)
    x = layers.Dropout(dropout_rate)(x)
    
    # Classifier head with 128 units (balance between capacity and overfitting)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=None  # No L2 regularization per pattern findings
    )(x)
    
    # Second dropout after dense activation (completes the "Aggressive Dropout Regime")
    x = layers.Dropout(dropout_rate)(x)
    
    # --- Output ---
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation("softmax", dtype="float32")(x)
    
    model = keras.Model(inputs, outputs)
    
    # --- Optimizer: Adam with LR ~0.002 ---
    # Pattern: LR ~0.002 enables fast convergence (mean_learning_speed 0.011) without instability
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model