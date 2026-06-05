def build_model(input_shape, num_classes, params: dict):
    from tensorflow.keras import layers, models, regularizers
    from tensorflow.keras.optimizers import Adam
    
    # Extract parameters with defaults based on design strategy
    num_conv_layers = params.get('num_conv_layers', 7)
    filters = params.get('filters', 32)  # Assumes single int or list of ints; if int, same for all layers
    kernel_size = params.get('kernel_size', 5)
    dense_units = params.get('dense_units', 64)
    dropout_rate = params.get('dropout_rate', 0.3)
    learning_rate = params.get('learning_rate', 1.5e-5)
    l2_reg = params.get('l2_reg', 0.01)  # Advanced regularization for memorization optimization
    batch_norm = params.get('batch_norm', True)  # Explore batch normalization for enhanced generalization
    # Multi-scale features: optional skip connections for simplicity (can be toggled)
    use_skip_connections = params.get('use_skip_connections', False)
    
    # Build the model
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    
    # Convolutional layers
    for i in range(num_conv_layers):
        if isinstance(filters, list):
            current_filters = filters[i] if i < len(filters) else filters[-1]
        else:
            current_filters = filters
        
        conv_layer = layers.Conv2D(
            current_filters, 
            kernel_size=kernel_size, 
            activation='relu', 
            padding='same',
            kernel_regularizer=regularizers.l2(l2_reg)
        )
        model.add(conv_layer)
        
        if batch_norm:
            model.add(layers.BatchNormalization())
        
        model.add(layers.Dropout(dropout_rate))
        
        # Optional pooling every 2 layers for multi-scale features
        if (i + 1) % 2 == 0:
            model.add(layers.MaxPooling2D((2, 2)))
        
        # Simple skip connection if enabled (residual for multi-scale)
        if use_skip_connections and i > 0:
            # For simplicity, assume same filters for skip
            skip = layers.Conv2D(current_filters, (1, 1), activation='relu')(model.layers[-3].output)  # Adjust index
            model.add(layers.Add()([model.layers[-1].output, skip]))  # This might need adjustment for Sequential
    
    # Flatten
    model.add(layers.Flatten())
    
    # Dense layers with regularization
    model.add(layers.Dense(
        dense_units, 
        activation='relu', 
        kernel_regularizer=regularizers.l2(l2_reg)
    ))
    model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Note: Early stopping (patience=15) should be incorporated in the training loop, e.g., via callbacks
    return model