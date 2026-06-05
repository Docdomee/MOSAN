def build_model(input_shape, num_classes, params: dict):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, Dropout, GlobalAveragePooling1D, Dense
    from tensorflow.keras.regularizers import l2
    
    # Extract hyperparameters from params dict, with defaults based on design strategy (optimized for speed and timeout prevention)
    num_conv_blocks = params.get('num_conv_blocks', 5)  # Reduced to 5 blocks for shallower model, faster training
    initial_filters = params.get('initial_filters', 128)  # Reduced starting filters to lower initial computation
    filter_multiplier = params.get('filter_multiplier', 1.2)  # Progressive increase, but capped below
    max_filters = params.get('max_filters', 512)  # Cap filters to prevent excessive growth
    kernel_size = params.get('kernel_size', 3)  # Small kernel for fine-grained features
    dropout_rate = params.get('dropout_rate', 0.3)  # Reduced for balance between regularization and speed
    dense_units = params.get('dense_units', 128)  # Smaller dense layer for generalization
    use_residual = params.get('use_residual', True)  # Enable residual connections for depth
    l2_weight_decay = params.get('l2_weight_decay', 1e-4)  # Small L2 regularization for stability
    
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Initial Conv1D block
    x = Conv1D(filters=initial_filters, kernel_size=kernel_size, padding='same', kernel_regularizer=l2(l2_weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Build stacked residual blocks for depth
    filters = initial_filters
    for block in range(num_conv_blocks):
        shortcut = x  # Residual connection
        
        # First conv in block
        x = Conv1D(filters=int(min(filters, max_filters)), kernel_size=kernel_size, padding='same', kernel_regularizer=l2(l2_weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        # Second conv in block
        x = Conv1D(filters=int(min(filters, max_filters)), kernel_size=kernel_size, padding='same', kernel_regularizer=l2(l2_weight_decay))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Add residual (shortcut) if use_residual and dimensions match
        if use_residual and shortcut.shape[-1] == x.shape[-1]:
            x = Add()([x, shortcut])
        x = Activation('relu')(x)
        
        # Increase filters progressively, capped
        filters = min(filters * filter_multiplier, max_filters)
    
    # Global pooling and dense layers
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_weight_decay))(x)
    x = Dropout(dropout_rate / 2)(x)  # Slightly lower dropout for dense
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model