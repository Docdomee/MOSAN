def build_model(input_shape, num_classes, params: dict):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    import numpy as np
    
    # --- Hyperparameters from Design Strategy ---
    # Critical shift: num_blocks + init_channels paradigm (not num_conv_layers)
    num_blocks = params.get("num_blocks", 3)  # Range: 2-4
    init_channels = params.get("init_channels", 64)  # Range: 48-96
    weight_decay = params.get("weight_decay", 0.005)  # Range: 0.005-0.01 (aggressive regularization)
    dropout_rate = params.get("dropout_rate", 0.6)  # Fixed: proven effective against overfitting
    learning_rate = params.get("learning_rate", 1e-4)  # Fixed: low LR for stable block training
    kernel_size = params.get("kernel_size", 3)
    dense_units = params.get("dense_units", 64)  # Reduced default to control parameter count
    use_data_augmentation = params.get("use_data_augmentation", False)
    use_residual = params.get("use_residual", True)  # Essential for this architecture
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # --- Data Augmentation ---
    if use_data_augmentation:
        x = layers.GaussianNoise(0.1)(x)
    
    # --- Initial Projection ---
    # Project input to init_channels dimension
    x = layers.Conv1D(
        init_channels,
        kernel_size=7,
        padding='same',
        kernel_regularizer=regularizers.l2(weight_decay),
        name='initial_conv'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # --- Residual Blocks Stack ---
    # Each block: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    # Downsample by 2x in each block except the last one to reduce sequence length
    for i in range(num_blocks):
        strides = 2 if i < num_blocks - 1 else 1
        
        shortcut = x
        
        # First conv in block (with downsampling if not last block)
        x = layers.Conv1D(
            init_channels,
            kernel_size,
            strides=strides,
            padding='same',
            kernel_regularizer=regularizers.l2(weight_decay),
            name=f'block_{i+1}_conv1'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second conv in block (no downsampling)
        x = layers.Conv1D(
            init_channels,
            kernel_size,
            padding='same',
            kernel_regularizer=regularizers.l2(weight_decay),
            name=f'block_{i+1}_conv2'
        )(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if dimensions changed (strides > 1 or channel mismatch)
        if strides != 1 or shortcut.shape[-1] != init_channels:
            shortcut = layers.Conv1D(
                init_channels,
                1,
                strides=strides,
                padding='same',
                kernel_regularizer=regularizers.l2(weight_decay),
                name=f'block_{i+1}_shortcut'
            )(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Residual connection
        if use_residual:
            x = layers.Add()([x, shortcut])
        
        x = layers.Activation('relu')(x)
    
    # --- Classification Head ---
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(
        dense_units,
        activation='relu',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation('softmax', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs)
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model