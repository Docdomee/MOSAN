def build_model(input_shape, num_classes, params: dict):
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, ReLU, MaxPooling3D, GlobalAveragePooling3D, Dense, Dropout, Reshape, Attention, Flatten
    from tensorflow.keras.models import Model
    
    # Extract params with defaults
    num_conv_layers = params.get('num_conv_layers', 8)
    filters = params.get('filters', 64)
    kernel_size = params.get('kernel_size', 3)
    dense_units = params.get('dense_units', 128)
    dropout_rate = params.get('dropout_rate', 0.35)
    learning_rate = params.get('learning_rate', 1.5e-5)
    weight_decay = params.get('weight_decay', 0.0)
    use_batch_norm = params.get('batch_norm', 1)
    
    # Input layer (expected shape: (time, height, width, channels))
    inputs = Input(shape=input_shape)
    
    # Depth-first 3D convolutional blocks
    x = inputs
    for i in range(num_conv_layers):
        x = Conv3D(filters=filters, kernel_size=(kernel_size, kernel_size, kernel_size), padding='same')(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        x = ReLU()(x)
        if (i + 1) % 2 == 0:
            x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    # Attention mechanism on 3D feature maps
    t, h, w, c = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
    x_reshaped = Reshape((t * h * w, c))(x)
    attn_output = Attention()([x_reshaped, x_reshaped])
    x = Reshape((t, h, w, c))(attn_output)
    
    # Global pooling and dense layers
    x = GlobalAveragePooling3D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile with optional weight decay
    if weight_decay > 0:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model