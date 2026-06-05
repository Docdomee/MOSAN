def build_model(input_shape, num_classes, params):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np
    
    # --- Architecture Constants (Motif-Optimized) ---
    # Strategy: Fixed template of 6 conv layers, 48 filters, kernel size 7
    num_conv_layers = 6
    num_filters = 48
    kernel_size = 7
    
    # --- Regularization Stack (Success Motifs) ---
    # Aggressive dropout 0.75-0.8 paired with weight decay 0.001-0.0025
    dropout_rate = params.get("dropout_rate", 0.8)
    weight_decay = params.get("weight_decay", 0.0015)  # Optimal middle of range
    
    # --- Optimization Parameters ---
    learning_rate = params.get("learning_rate", 0.0001)
    
    # --- Regularizer ---
    reg = keras.regularizers.l2(weight_decay)
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # --- Data Augmentation ---
    if params.get("use_data_augmentation"):
        x = layers.GaussianNoise(0.1)(x)
    
    # --- Convolutional Tower (6 layers with Residual Connections) ---
    for i in range(num_conv_layers):
        # Dilated convolutions in later layers (indices 3,4,5) to increase 
        # receptive field without adding parameters (Spectral-Specific Enhancement)
        dilation_rate = 2 if i >= 3 else 1
        
        # Residual shortcut
        shortcut = x
        
        # Main convolutional path
        x = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            dilation_rate=dilation_rate,
            kernel_regularizer=reg,
            kernel_initializer='he_normal',
            use_bias=False  # BatchNorm handles bias
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Residual connection (always enabled per design strategy)
        if shortcut.shape[-1] != num_filters:
            shortcut = layers.Conv1D(
                num_filters, 
                1, 
                padding="same",
                kernel_regularizer=reg,
                kernel_initializer='he_normal',
                use_bias=False
            )(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        
        # Downsampling
        x = layers.MaxPooling1D(pool_size=2)(x)
    
    # --- Spectral Attention Mechanism (SE-Block) ---
    # Focus on discriminative wavenumbers/channels
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(num_filters // 4, activation='relu', kernel_regularizer=reg)(se)
    se = layers.Dense(num_filters, activation='sigmoid', kernel_regularizer=reg)(se)
    se = layers.Reshape((1, num_filters))(se)
    x = layers.Multiply()([x, se])
    
    # --- Classification Head ---
    # Global Average Pooling to reduce parameter count (avoiding 3.9M param overfitting)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layer with 48 units as per motif analysis
    x = layers.Dense(
        units=params.get("dense_units", 48),
        activation="relu",
        kernel_regularizer=reg,
        kernel_initializer='he_normal'
    )(x)
    
    # Aggressive dropout regime (0.75-0.8) for best generalization
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    x = layers.Dense(num_classes, kernel_regularizer=reg)(x)
    outputs = layers.Activation("softmax", dtype="float32")(x)
    
    model = keras.Model(inputs, outputs)
    
    # --- Optimizer ---
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model