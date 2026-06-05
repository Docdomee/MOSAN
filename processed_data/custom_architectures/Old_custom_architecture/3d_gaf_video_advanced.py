def build_model(input_shape, num_classes, params: dict):
    from keras import layers
    from tensorflow import keras

    # Adjust input_shape for validation (3D dummy) to 5D for 3D_GAF_VIDEO
    if len(input_shape) == 3:
        input_shape = (12, 24, 24, input_shape[-1])

    # Extract parameters with defaults
    num_conv_layers = params.get("num_conv_layers", 4)
    initial_filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 3)
    dense_units = params.get("dense_units", 256)
    dropout_rate = params.get("dropout_rate", 0.3)
    learning_rate = params.get("learning_rate", 0.001)

    # Functional API
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Convolutional layers with progressive filter doubling and batch norm
    filters = initial_filters
    for i in range(num_conv_layers):
        x = layers.Conv3D(filters, (kernel_size, kernel_size, kernel_size), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        if i < num_conv_layers - 1:
            x = layers.MaxPooling3D((2, 2, 2))(x)
        filters *= 2  # Progressive doubling

    # Global average pooling
    x = layers.GlobalAveragePooling3D()(x)

    # Dense layers with dropout
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Simple multi-head attention (reshaped for features)
    attention_input = layers.Reshape((1, dense_units))(x)
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=dense_units // 4)(attention_input, attention_input)
    attention = layers.Flatten()(attention)

    # Final dense layer
    outputs = layers.Dense(num_classes, activation="softmax")(attention)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
