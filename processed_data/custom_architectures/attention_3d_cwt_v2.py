def build_model(input_shape, num_classes, params):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    filters = params.get('filters', 48)
    dropout_rate = params.get('dropout_rate', 0.45)
    attention_dim = params.get('attention_dim', 128)
    num_heads = params.get('num_heads', 4)
    learning_rate = params.get('lr', 0.0004)
    weight_decay = params.get('weight_decay', 3e-5)
    num_conv_layers = params.get('num_conv_layers', 3)
    dense_units = params.get('dense_units', 128)
    batch_norm = params.get('batch_norm', 1)

    key_dim = attention_dim // num_heads

    inputs = keras.Input(shape=input_shape)
    x = inputs

    # 3D Convolutional Backbone
    for i in range(num_conv_layers):
        x = layers.Conv3D(
            filters=filters * (2 ** min(i, 3)),
            kernel_size=3,
            padding='same',
            kernel_regularizer=regularizers.l2(weight_decay),
        )(x)
        if batch_norm == 1:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.SpatialDropout3D(dropout_rate * 0.5)(x)
        if i < num_conv_layers - 1:
            x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)

    # Global Average Pooling -> (batch, channels)
    x = layers.GlobalAveragePooling3D()(x)
    
    # Project to attention_dim and reshape for sequence attention
    x = layers.Dense(attention_dim, activation='relu')(x)
    # Reshape to (batch, seq_len=1, feature_dim)
    x = layers.Reshape((1, attention_dim))(x)

    # Multi-head self-attention
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate * 0.8)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # Feed-forward network
    ff = layers.Dense(attention_dim * 2, activation='relu')(x)
    ff = layers.Dropout(dropout_rate)(ff)
    ff = layers.Dense(attention_dim)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)

    # Pool and classify
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units // 2, activation='relu', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='attention_3d_cwt_v2')
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
