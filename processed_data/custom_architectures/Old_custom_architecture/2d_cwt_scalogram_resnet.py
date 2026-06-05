def build_model(input_shape, num_classes, params: dict):
    from keras import layers
    from tensorflow import keras

    # Extract hyperparameters from params dict, with defaults based on historical insights
    num_conv_layers = params.get("num_conv_layers", 4)
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 5)
    dense_units = params.get("dense_units", 256)
    dropout_rate = params.get("dropout_rate", 0.32)
    learning_rate = params.get("learning_rate", 0.007)

    inputs = keras.Input(shape=input_shape)

    # Initial conv layer
    x = layers.Conv2D(filters, (kernel_size, kernel_size), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Residual blocks for subsequent layers
    for i in range(1, num_conv_layers):
        # Adjust filters progressively (doubling every two layers, inspired by historical patterns)
        current_filters = filters * (2 ** ((i - 1) // 2))
        residual = x  # Store input for residual connection

        x = layers.Conv2D(current_filters, (kernel_size, kernel_size), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(current_filters, (kernel_size, kernel_size), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)

        # Adjust residual if filter count changes
        if residual.shape[-1] != current_filters:
            residual = layers.Conv2D(current_filters, (1, 1), padding="same")(residual)

        x = layers.Add()([x, residual])  # Residual connection
        x = layers.MaxPooling2D((2, 2))(x)

    # Global pooling and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    # Compile with optimizer based on historical learning rate insights
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model
