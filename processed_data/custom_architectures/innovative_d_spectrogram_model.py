def build_model(input_shape, num_classes, params: dict):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    import numpy as np

    # Extract hyperparameters from params dict, with defaults based on successful motifs
    num_conv_layers = params.get('num_conv_layers', 8)  # 7-8 as baseline
    filters = params.get('filters', 128)  # 128-192
    dropout_rate = params.get('dropout_rate', 0.65)  # 0.58-0.69
    weight_decay = params.get('weight_decay', 0.004)  # 0.003-0.01
    learning_rate = params.get('learning_rate', 1.5e-5)  # 1e-5 to 2e-5
    use_batch_norm = params.get('use_batch_norm', True)  # Recommended exploration
    # Additional params for innovation
    use_attention = params.get('use_attention', True)  # Enable attention mechanism
    multi_scale = params.get('multi_scale', True)  # Enable multi-scale processing

    # Define a residual block with multi-scale convolutions and optional attention
    def residual_multi_scale_block(x, filters, stride=1, use_attention=False):
        shortcut = x
        
        # Multi-scale branch: different kernel sizes for spectral feature extraction
        branch1 = layers.Conv2D(filters//4, (1,1), strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        branch1 = layers.BatchNormalization()(branch1) if use_batch_norm else branch1
        branch1 = layers.ReLU()(branch1)
        
        branch2 = layers.Conv2D(filters//4, (3,3), strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        branch2 = layers.BatchNormalization()(branch2) if use_batch_norm else branch2
        branch2 = layers.ReLU()(branch2)
        
        branch3 = layers.Conv2D(filters//4, (5,5), strides=stride, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        branch3 = layers.BatchNormalization()(branch3) if use_batch_norm else branch3
        branch3 = layers.ReLU()(branch3)
        
        # Concatenate multi-scale features
        x = layers.Concatenate()([branch1, branch2, branch3])
        
        # Attention mechanism: Squeeze-and-Excitation (SE) block
        if use_attention:
            channels = x.shape[-1]
            se_shape = (1, 1, channels)
            squeeze = layers.GlobalAveragePooling2D()(x)
            excitation = layers.Dense(channels // 16, activation='relu')(squeeze)
            excitation = layers.Dense(channels, activation='sigmoid')(excitation)
            excitation = layers.Reshape(se_shape)(excitation)
            x = layers.Multiply()([x, excitation])
        
        # Final conv to match filters
        x = layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        x = layers.BatchNormalization()(x) if use_batch_norm else x
        
        # Residual connection
        if stride > 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1,1), strides=stride, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(shortcut)
            shortcut = layers.BatchNormalization()(shortcut) if use_batch_norm else shortcut
        
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    # Build the model
    inputs = layers.Input(shape=input_shape)
    
    # Initial conv layer
    x = layers.Conv2D(filters, (7,7), strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x) if use_batch_norm else x
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)
    
    # Stacked residual multi-scale blocks
    for i in range(num_conv_layers - 1):  # Adjust for initial conv
        stride = 2 if i in [2, 4] else 1  # Downsample at certain layers for deeper features
        x = residual_multi_scale_block(x, filters, stride=stride, use_attention=use_attention and (i % 2 == 0))  # Apply attention every other block
    
    # Global pooling and classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    # Compile with optimized learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model