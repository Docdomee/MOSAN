def build_model(input_shape, num_classes, params):
    """
    Optimized 7-Layer Deep CNN with Squeeze-and-Excitation Blocks and Residual Connections.
    
    Architecture improvements based on Pattern Analysis:
    - 7 Convolutional layers with residual connections and SE attention
    - Balanced regularization: Moderate dropout (0.40-0.50) + L2 (1e-4 to 5e-4)
    - Learning rate sweet spot: 4e-5 to 9e-5
    - Batch normalization for training stability
    
    Exploration vector supported: Higher L2 (5e-4 to 8e-4) with lower dropout (0.35-0.45)
    via params dictionary.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # --- Hyperparameters from Design Strategy ---
    # Learning rate: Prioritize [4e-5, 1e-4] range (avoid <2e-5)
    learning_rate = params.get("learning_rate", 6.5e-5)
    
    # Regularization: Balanced stack (avoid high dropout >0.55 + high L2 >0.0006)
    dropout_rate = params.get("dropout_rate", 0.45)  # Working range: 0.40-0.50
    l2_lambda = params.get("l2_reg", 0.0003)  # Working range: 0.0001-0.0005
    
    regularizer = keras.regularizers.l2(l2_lambda)

    def se_block(inputs, reduction=16):
        """Squeeze-and-Excitation block for channel attention."""
        filters = inputs.shape[-1]
        se = layers.GlobalAveragePooling2D()(inputs)
        se = layers.Reshape((1, 1, filters))(se)
        se = layers.Dense(
            filters // reduction, 
            activation='relu', 
            use_bias=False,
            kernel_regularizer=regularizer
        )(se)
        se = layers.Dense(
            filters, 
            activation='sigmoid', 
            use_bias=False,
            kernel_regularizer=regularizer
        )(se)
        return layers.multiply([inputs, se])

    def conv_bn(x, filters, kernel_size=3, stride=1, activation=True):
        """Conv2D + BatchNorm + optional ReLU."""
        x = layers.Conv2D(
            filters, 
            kernel_size, 
            strides=stride, 
            padding='same',
            kernel_regularizer=regularizer,
            kernel_initializer='he_normal'
        )(x)
        x = layers.BatchNormalization()(x)
        if activation:
            x = layers.ReLU()(x)
        return x

    # --- Build Architecture ---
    inputs = keras.Input(shape=input_shape)
    
    # Layer 1: Initial stem (7x7 conv, stride 2)
    x = conv_bn(inputs, 64, kernel_size=7, stride=2)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # Layer 2-3: Residual Block 1 (64 filters)
    shortcut = x
    x = conv_bn(x, 64, kernel_size=3, stride=1)
    x = conv_bn(x, 64, kernel_size=3, stride=1, activation=False)  # No activation before addition
    x = se_block(x)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    
    # Layer 4-5: Residual Block 2 (128 filters, stride 2)
    shortcut = conv_bn(x, 128, kernel_size=1, stride=2, activation=False)  # Projection shortcut
    x = conv_bn(x, 128, kernel_size=3, stride=2)
    x = conv_bn(x, 128, kernel_size=3, stride=1, activation=False)
    x = se_block(x)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    
    # Layer 6-7: Residual Block 3 (256 filters, stride 2)
    shortcut = conv_bn(x, 256, kernel_size=1, stride=2, activation=False)  # Projection shortcut
    x = conv_bn(x, 256, kernel_size=3, stride=2)
    x = conv_bn(x, 256, kernel_size=3, stride=1, activation=False)
    x = se_block(x)
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    
    # Global Average Pooling and Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(
        num_classes, 
        activation='softmax',
        kernel_regularizer=regularizer
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile with Adam optimizer using the specified learning rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model