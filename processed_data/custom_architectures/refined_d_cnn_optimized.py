def build_model(input_shape, num_classes, params: dict):
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling1D
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    
    # Defaults defined for reproducibility
    def get_defaults():
        return {
            'num_conv_layers': 14,
            'initial_filters': 64,
            'kernel_size': 5,
            'dense_units': 512,
            'dropout_rate': 0.55,
            'learning_rate': 0.0006,
            'pool_size': 2
        }

    # Extract hyperparameters from params dict, with defaults based on Design Strategy
    defaults = get_defaults()
    num_conv_layers = params.get('num_conv_layers', defaults['num_conv_layers'])  # 12-15 range, choosing 14 as balance
    initial_filters = params.get('initial_filters', defaults['initial_filters'])  # Starting filters, will increase
    kernel_size = params.get('kernel_size', defaults['kernel_size'])  # 5 or 7, choosing 5 for balance per strategy
    dense_units = params.get('dense_units', defaults['dense_units'])  # 256-512, choosing 512 for capacity
    dropout_rate = params.get('dropout_rate', defaults['dropout_rate'])  # 0.3-0.65, choosing 0.55 for stability in deeper model
    learning_rate = params.get('learning_rate', defaults['learning_rate'])  # 0.0004-0.0008, choosing 0.0006 for stability
    pool_size = params.get('pool_size', defaults['pool_size'])
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Convolutional layers: 14 layers, with increasing filters and interspersed pooling/dropout
    x = inputs
    filters = initial_filters
    for i in range(num_conv_layers):
        x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)  # Added for stability, common in deep nets
        if (i + 1) % 2 == 0:  # Pool every two layers to control size
            x = MaxPooling1D(pool_size=pool_size)(x)
        filters = min(filters * 2, 256)  # Cap at 256 to keep params moderate (7-10M target)
    
    # Global Average Pooling to reduce params before dense
    x = GlobalAveragePooling1D()(x)
    
    # Dropout for regularization
    x = Dropout(dropout_rate)(x)
    
    # Dense layers: Two dense layers with dropout in between for capacity and prevention of overfitting
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units // 2, activation='relu')(x)  # Reduced units for output layer
    x = Dropout(dropout_rate * 0.8)(x)  # Slightly lower dropout for penultimate layer
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Build model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile with Adam optimizer and low LR for stability
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model