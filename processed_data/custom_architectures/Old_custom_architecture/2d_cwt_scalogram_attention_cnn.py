def build_model(input_shape, num_classes, params: dict):
    from keras import layers
    from tensorflow import keras

    # Extract params
    num_conv_layers = params.get("num_conv_layers", 4)
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 7)
    dense_units = params.get("dense_units", 256)
    dropout_rate = params.get("dropout_rate", 0.32)
    learning_rate = params.get("learning_rate", 0.008)

    # Build model
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Convolutional layers with attention
    for i in range(num_conv_layers):
        # Conv layer
        x = layers.Conv2D(filters * (2**i), (kernel_size, kernel_size), activation="relu", padding="same")(x)
        # Batch norm for stability
        x = layers.BatchNormalization()(x)
        # Max pooling
        x = layers.MaxPooling2D((2, 2))(x)
        # Channel attention module (squeeze-excitation like)
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Dense(filters * (2**i) // 16, activation="relu")(se)
        se = layers.Dense(filters * (2**i), activation="sigmoid")(se)
        x = layers.Multiply()([x, layers.Reshape((1, 1, filters * (2**i)))(se)])

    # Flatten
    x = layers.Flatten()(x)
    # Dense layers
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Compile model
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
