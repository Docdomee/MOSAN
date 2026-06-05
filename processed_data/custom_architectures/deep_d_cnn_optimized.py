def build_model(input_shape, num_classes, params: dict):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Pattern Scout Optimal Configuration
    # Depth > Width: 6 layers is the sweet spot (vs 5)
    max_layers = int(np.log2(input_shape[0])) if input_shape[0] > 0 else 1
    requested_layers = params.get("num_conv_layers", 6)
    num_conv_layers = min(requested_layers, max_layers)
    
    if num_conv_layers < 6:
        print(f"[Architect Warning] Requested 6 layers, but input supports only {num_conv_layers}. Consider padding input for optimal performance.")
    
    # Pattern Scout Hyperparameters (Anti-overfitting configuration)
    filters = params.get("filters", 48)          # 48 filters: better generalization than 64
    kernel_size = params.get("kernel_size", 7)   # Fixed receptive field for spectral peaks
    dense_units = params.get("dense_units", 48)  # Bottleneck: 48 units prevents overfitting
    dropout_rate = params.get("dropout_rate", 0.75)  # Aggressive regularization regime
    weight_decay = params.get("weight_decay", 0.0025) # L2 sweet spot (never < 0.001)
    learning_rate = params.get("learning_rate", 0.00013) # Conservative LR for stability
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Data Augmentation (optional Gaussian noise)
    if params.get("use_data_augmentation"):
        x = layers.GaussianNoise(0.1)(x)
    
    # Regularization with Pattern Scout weight decay
    reg = keras.regularizers.l2(weight_decay)
    
    # Deep 6-Layer Architecture with Residual Connections
    for i in range(num_conv_layers):
        shortcut = x
        
        # Fixed 48 filters per layer (no width expansion) to maintain bottleneck
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=reg,
            kernel_initializer="he_normal"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Residual connection for gradient flow in deep network
        if params.get("use_residual", True):
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv1D(
                    filters=filters,
                    kernel_size=1,
                    padding="same",
                    kernel_regularizer=reg
                )(shortcut)
            x = layers.Add()([x, shortcut])
        
        # Aggressive downsampling
        x = layers.MaxPooling1D(2)(x)
    
    # Global Average Pooling (feature compression)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Bottleneck Dense Layer (48 units)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=reg,
        kernel_initializer="he_normal"
    )(x)
    
    # High dropout for regularization (0.70-0.75 range)
    x = layers.Dropout(dropout_rate)(x)
    
    # Classification head
    x = layers.Dense(num_classes, kernel_regularizer=reg)(x)
    outputs = layers.Activation("softmax", dtype="float32")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Conservative optimizer settings
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model