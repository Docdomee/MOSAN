def build_model(input_shape, num_classes, params: dict):
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling3D, Reshape, LayerNormalization, MultiHeadAttention, Add
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    
    defaults = {
        'num_conv_layers': 4,
        'filters': 64,
        'kernel_size': 3,
        'dense_units': 256,
        'dropout_rate': 0.4,
        'learning_rate': 0.0005,
        'weight_decay': 0.0001,
        'batch_norm': 1,
        'num_attention_heads': 4,
        'attention_dim': 128
    }
    
    num_conv_layers = params.get('num_conv_layers', defaults['num_conv_layers'])
    filters = params.get('filters', defaults['filters'])
    kernel_size = params.get('kernel_size', defaults['kernel_size'])
    dense_units = params.get('dense_units', defaults['dense_units'])
    dropout_rate = params.get('dropout_rate', defaults['dropout_rate'])
    learning_rate = params.get('learning_rate', defaults['learning_rate'])
    weight_decay = params.get('weight_decay', defaults['weight_decay'])
    batch_norm = params.get('batch_norm', defaults['batch_norm'])
    num_attention_heads = params.get('num_attention_heads', defaults['num_attention_heads'])
    attention_dim = params.get('attention_dim', defaults['attention_dim'])
    
    inputs = Input(shape=input_shape)
    x = inputs
    
    # 3D Convolutional backbone with increasing filters
    for i in range(num_conv_layers):
        x = Conv3D(filters=filters * (2 ** min(i, 3)), kernel_size=kernel_size, padding='same', activation='relu')(x)
        if batch_norm == 1:
            x = BatchNormalization()(x)
        x = MaxPooling3D(pool_size=(1, 2, 2) if i == 0 else 2)(x)
        x = Dropout(dropout_rate * 0.5)(x)
    
    # Self-attention block to suppress session noise
    # Project to attention dimension
    attn_input = GlobalAveragePooling3D()(x)
    if len(attn_input.shape) == 2:
        # Reshape to sequence for attention: (batch, seq_len=1, features)
        attn_input = Reshape((1, attn_input.shape[-1]))(attn_input)
    
    # Multi-head self-attention
    attn_output = MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=attention_dim // num_attention_heads
    )(attn_input, attn_input)
    
    # Skip connection
    attn_output = Add()([attn_input, attn_output])
    x = LayerNormalization()(attn_output)
    x = Flatten()(x)
    
    # Dense classifier head
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units // 2, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model