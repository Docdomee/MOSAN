def build_model(input_shape, num_classes, params):
    """
    Advanced 1D CNN optimized for SERS spectral data.
    
    Evolution from base 1D_CNN incorporating:
    - Fixed 6-layer conv stack (48 filters, kernel_size=7) per Pattern Scout analysis
    - Aggressive regularization (dropout=0.8, weight_decay=0.001)
    - Spectral attention mechanism for wavenumber region highlighting
    - Residual connections to stabilize deep stack with high dropout
    - Flatten-based feature aggregation (proven superior to global pooling alone)
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np

    # --- Pattern Scout Optimized Defaults ---
    # These values are non-negotiable based on trial analysis
    num_conv_layers = params.get("num_conv_layers", 6)  # Deep stack for spectral feature extraction
    filters = params.get("filters", 48)  # Fixed 48 filters per layer
    kernel_size = params.get("kernel_size", 7)  # Large receptive field for spectral patterns
    dense_units = params.get("dense_units", 48)  # Non-negotiable dense projection
    dropout_rate = params.get("dropout_rate", 0.8)  # Aggressive dropout to prevent overfitting
    weight_decay = params.get("weight_decay", 0.001)  # Optimal L2 sweet spot
    use_residual = params.get("use_residual", True)  # Enable gradient flow in deep stack
    use_spectral_attention = params.get("use_spectral_attention", True)  # Highlight discriminative wavenumbers
    
    # --- Defensive Layer Calculation ---
    max_layers = int(np.log2(input_shape[0])) if input_shape[0] > 0 else 1
    actual_num_conv = min(num_conv_layers, max_layers)
    if actual_num_conv < num_conv_layers:
        print(f"[Advanced 1D CNN] Warning: Reduced conv layers from {num_conv_layers} to {actual_num_conv} due to input size.")
    
    # --- Regularizer ---
    reg = keras.regularizers.l2(weight_decay)
    
    # --- Input ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # --- Data Augmentation ---
    if params.get("use_data_augmentation", False):
        x = layers.GaussianNoise(0.1)(x)
    
    # --- Convolutional Stack with Residuals ---
    for i in range(actual_num_conv):
        shortcut = x
        
        # Conv block
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=reg,
            kernel_initializer="he_normal"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Residual connection (critical for 6-layer depth with high dropout)
        if use_residual:
            # Match channel dimensions if necessary
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv1D(
                    filters, 
                    kernel_size=1, 
                    padding="same",
                    kernel_regularizer=reg
                )(shortcut)
            x = layers.Add()([x, shortcut])
        
        # Downsampling
        x = layers.MaxPooling1D(pool_size=2)(x)
    
    # --- Spectral Attention Mechanism ---
    # Highlights discriminative wavenumber regions after feature extraction
    if use_spectral_attention:
        # Compute attention scores across spectral dimension (axis=1)
        # Input shape: (batch, time_steps, filters)
        attn = layers.Conv1D(1, kernel_size=1, padding="same")(x)  # (batch, time_steps, 1)
        attn = layers.Flatten()(attn)  # (batch, time_steps)
        attn = layers.Activation("softmax")(attn)  # Softmax over spectral dimension
        attn = layers.Reshape((-1, 1))(attn)  # (batch, time_steps, 1)
        
        # Apply attention weights to features
        x = layers.Multiply()([x, attn])  # Broadcast multiply: features * attention_weights
    
    # --- Feature Aggregation ---
    # Flatten preserves spatial-spectral relationships better than global pooling for this data
    x = layers.Flatten()(x)
    
    # --- Classification Head with Aggressive Regularization ---
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=reg,
        kernel_initializer="he_normal"
    )(x)
    x = layers.Dropout(dropout_rate)(x)  # High dropout (0.8) proven optimal
    
    # Output layer
    x = layers.Dense(num_classes, kernel_regularizer=reg)(x)
    outputs = layers.Activation("softmax", dtype="float32")(x)
    
    # --- Model Compilation ---
    model = keras.Model(inputs, outputs)
    
    optimizer = keras.optimizers.Adam(
        learning_rate=params.get("learning_rate", 1e-3)
    )
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model