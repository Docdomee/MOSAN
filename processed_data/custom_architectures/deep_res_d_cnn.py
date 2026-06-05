def build_model(input_shape, num_classes, params):
    """
    Deep 8-Layer Residual 1D-CNN Architecture.
    
    Implements the Pattern Scout prescription:
    - 7-8 Convolutional layers with kernel_size=3 exclusively
    - Conservative filter strategy (32-64) prioritizing depth over width
    - Aggressive regularization (dropout 0.58-0.67) with moderate dense layers (256-512)
    - Conservative learning rates (1e-5 to 1.5e-4)
    - Residual connections to enable deep gradient flow
    """
    import numpy as np
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # --- Architecture Hyperparameters (Pattern Scout Optimized) ---
    num_conv_layers = params.get("num_conv_layers", 8)  # Mandatory depth: 7-8 layers
    kernel_size = 3  # Fixed per Pattern Scout findings
    dropout_rate = params.get("dropout_rate", 0.62)  # Target 0.60-0.65 range
    dense_units = params.get("dense_units", 384)  # 256-512 range, avoid 1024
    learning_rate = params.get("learning_rate", 8e-5)  # Conservative: 5e-5 to 1.4e-4
    use_residual = params.get("use_residual", True)
    use_augmentation = params.get("use_data_augmentation", False)
    l2_reg = params.get("l2_reg", 1e-4)
    
    # --- Input ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # --- Data Augmentation ---
    if use_augmentation:
        x = layers.GaussianNoise(0.1)(x)
    
    # --- Regularizer ---
    reg = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    
    # --- Deep Convolutional Stack (7-8 Layers) ---
    current_length = input_shape[0] if isinstance(input_shape, (list, tuple)) else input_shape
    
    for i in range(num_conv_layers):
        # Filter strategy: 32-64 range for 8-layer depth (slow growth, prefer depth)
        if i < 4:
            num_filters = 32
        else:
            num_filters = 64
        
        # Residual shortcut
        shortcut = x
        
        # Convolutional Block: Conv -> BN -> ReLU
        x = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=reg,
            kernel_initializer="he_normal",
            name=f"conv_{i+1}"
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)
        
        # Residual Connection (Projection if dimensions mismatch)
        if use_residual:
            if shortcut.shape[-1] != num_filters:
                shortcut = layers.Conv1D(
                    filters=num_filters,
                    kernel_size=1,
                    padding="same",
                    kernel_initializer="he_normal",
                    name=f"proj_{i+1}"
                )(shortcut)
            x = layers.Add(name=f"res_{i+1}")([x, shortcut])
        
        # Progressive Pooling: Only pool if sequence length > 1 to allow 8 layers on 80-length data
        if current_length > 1:
            x = layers.MaxPooling1D(pool_size=2, name=f"pool_{i+1}")(x)
            current_length = current_length // 2
    
    # --- Global Feature Aggregation ---
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    
    # --- Bottleneck Dense Layer (Efficient Design) ---
    x = layers.Dense(
        units=dense_units,
        activation="relu",
        kernel_regularizer=reg,
        kernel_initializer="he_normal",
        name="dense_bottleneck"
    )(x)
    x = layers.Dropout(rate=dropout_rate, name="dropout")(x)
    
    # --- Classification Head ---
    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        kernel_initializer="he_normal",
        name="predictions"
    )(x)
    
    # --- Model Compilation with Conservative Optimization ---
    model = keras.Model(inputs=inputs, outputs=outputs, name="deep_res_1d_cnn")
    
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model