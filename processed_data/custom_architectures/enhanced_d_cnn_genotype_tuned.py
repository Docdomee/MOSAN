def build_model(input_shape, num_classes, params: dict):
    """
    Builds an evolved 1D CNN model based on the genotype 1767638394, enhanced with SOTA techniques
    from NAS (e.g., DARTS-inspired ops) and targeted hyperparameter tuning for robustness.
    
    Evolutions:
    - Retained core ops: dil_conv_5x5, skip_connect for depth and gradient flow.
    - Added sep_conv_3x3 in normal cells for finer-grained features, reducing kernel size reliance.
    - Increased drop_path to 0.2 in reduction cells for better generalization.
    - Tuned L2 regularization to 5e-5 to balance with dropout.
    - Hyperparams pre-set in params dict for optimal ranges (e.g., LR 2e-4, dropout 0.25, batch 16, channels 56, layers 11).
    - Incorporated adaptive pooling in some cells for efficiency (SOTA from modern NAS like AutoML).
    
    Assumes params contains: 'learning_rate', 'dropout', 'init_channels', 'layers', 'drop_path', 'l2_reg', etc.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers
    import numpy as np
    
    # Extract params with defaults based on strategy
    learning_rate = params.get('learning_rate', 2e-4)
    dropout_rate = params.get('dropout', 0.25)
    init_channels = params.get('init_channels', 56)
    num_layers = params.get('layers', 11)
    drop_path_rate = params.get('drop_path', 0.1)  # Will be increased in reductions
    l2_reg = params.get('l2_reg', 5e-5)
    batch_size = params.get('batch_size', 16)  # For reference, though not used in build
    
    def conv_1d(x, filters, kernel_size, strides=1, dilation=1, l2_reg=l2_reg):
        return layers.Conv1D(filters, kernel_size, strides=strides, dilation_rate=dilation,
                             padding='same', kernel_regularizer=regularizers.l2(l2_reg))(x)
    
    def sep_conv_1d(x, filters, kernel_size, strides=1, l2_reg=l2_reg):
        return layers.SeparableConv1D(filters, kernel_size, strides=strides, padding='same',
                                       depthwise_regularizer=regularizers.l2(l2_reg),
                                       pointwise_regularizer=regularizers.l2(l2_reg))(x)
    
    def dil_conv_1d(x, filters, kernel_size, dilation, strides=1, l2_reg=l2_reg):
        return conv_1d(x, filters, kernel_size, dilation=dilation, strides=strides, l2_reg=l2_reg)
    
    def skip_connect(x, filters, strides=1):
        if x.shape[-1] != filters or strides > 1:
            x = conv_1d(x, filters, 1, strides=strides)  # 1x1 conv for channel adjustment and stride
        return x
    
    def avg_pool_1d(x, pool_size):
        return layers.AveragePooling1D(pool_size, strides=pool_size, padding='same')(x)
    
    def max_pool_1d(x, pool_size):
        return layers.MaxPooling1D(pool_size, strides=pool_size, padding='same')(x)
    
    def adaptive_avg_pool_1d(x, output_size):
        # Custom adaptive pooling for 1D
        return layers.GlobalAveragePooling1D()(x) if output_size == 1 else layers.AveragePooling1D(
            pool_size=x.shape[1] // output_size, strides=x.shape[1] // output_size, padding='same')(x)
    
    def drop_path(x, drop_prob=0.0):
        if drop_prob == 0.0:
            return x
        keep_prob = 1 - drop_prob
        shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
        random_tensor = tf.floor(random_tensor)
        return x / keep_prob * random_tensor
    
    # Genotype-inspired cell structure (evolved)
    def normal_cell(x, channels):
        # Multi-path with concat [2,3,4,5] as in original, but added sep_conv_3x3
        path1 = dil_conv_1d(x, channels, 5, dilation=2)
        path2 = sep_conv_1d(x, channels, 3)  # New addition for finer features
        path3 = skip_connect(x, channels)
        # Fixed: Use strides=1 to preserve sequence length in normal cells (no resolution reduction)
        path4 = layers.AveragePooling1D(3, strides=1, padding='same')(x)
        path5 = layers.MaxPooling1D(3, strides=1, padding='same')(x)
        return layers.Concatenate()([path1, path2, path3, path4, path5])
    
    def reduction_cell(x, channels):
        # Reduction with increased drop_path for generalization, ensuring all paths reduce sequence length by factor of 2 for matching shapes
        path1 = max_pool_1d(x, 2)  # pool_size=2, strides=2 for reduction
        path2 = sep_conv_1d(x, channels, 5, strides=2)
        path3 = skip_connect(x, channels, strides=2)  # Added strides for reduction
        path4 = dil_conv_1d(x, channels, 3, dilation=2, strides=2)  # Added strides for reduction
        path5 = adaptive_avg_pool_1d(x, x.shape[1] // 2)
        concat = layers.Concatenate()([path1, path2, path3, path4, path5])
        return drop_path(concat, drop_path_rate * 2)  # Increased drop_path
    
    # Build model
    inputs = layers.Input(shape=input_shape)
    x = conv_1d(inputs, init_channels, 3)  # Initial conv
    
    for layer in range(num_layers):
        if layer % 3 == 2 and layer != 0:  # Reduction at layers/3 and 2*layers/3
            channels = init_channels * (2 ** (layer // 3))
            x = reduction_cell(x, channels)
        else:
            x = normal_cell(x, init_channels)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Global pooling and output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    # Compile with tuned optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model