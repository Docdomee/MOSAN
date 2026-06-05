def build_model(input_shape, num_classes, params: dict):
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Input, Conv1D, BatchNormalization, Activation, Dropout, Add, GlobalAveragePooling1D, Dense, Attention
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2
    
    # Extract hyperparameters from params dict, with defaults based on design strategy
    num_conv_layers = params.get('num_conv_layers', 8)  # Prioritize depth: 6-10 layers
    filters = params.get('filters', 128)  # Moderate: 64-256
    kernel_size = params.get('kernel_size', 3)  # Small: 3, test 5 if needed
    dense_units = params.get('dense_units', 64)  # Small: 64-128 max
    dropout_rate = params.get('dropout_rate', 0.4)  # Moderate: 0.3-0.5
    l2_reg = params.get('l2_reg', 1e-4)  # Light L2 regularization to control overfitting
    
    inputs = Input(shape=input_shape)
    
    # Initial Conv block
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Subsequent Conv blocks with residual connections
    for i in range(num_conv_layers - 1):
        # Residual connection
        residual = x
        
        # Main path
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        # Add attention-inspired mechanism: simple multi-head attention for feature enhancement
        # Using Keras Attention layer for self-attention on 1D features
        attention_out = Attention(use_scale=True)([x, x])  # Self-attention to capture temporal dependencies
        x = Add()([x, attention_out])  # Integrate attention output
        
        # Residual add (if shapes match)
        if residual.shape[-1] == x.shape[-1]:
            x = Add()([x, residual])
    
    # Global Average Pooling for efficiency
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    
    # Dense layer
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    
    # Output
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model