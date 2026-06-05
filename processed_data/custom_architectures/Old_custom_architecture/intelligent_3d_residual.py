def build_model(input_shape, num_classes, params: dict):
    from keras import layers
    from tensorflow import keras

    inputs = keras.Input(shape=input_shape)

    # Handle 4D validation inputs vs 5D runtime inputs
    # If input is 4D (validation dummy), reshape to 5D by adding temporal dimension
    # If input is already 5D (runtime data), use as-is
    if len(input_shape) == 3:  # 4D input (batch, height, width, channels)
        x = layers.Reshape((1, input_shape[0], input_shape[1], input_shape[2]))(inputs)
    else:  # 5D input (batch, segments, height, width, channels)
        x = inputs

    # Initial convolution with batch normalization
    x = layers.Conv3D(
        filters=params.get("initial_filters", 32),
        kernel_size=params.get("initial_kernel", 3),
        padding="same",
        activation="relu",
    )(x)
    x = layers.BatchNormalization()(x)

    # Residual blocks with progressive filter scaling
    num_blocks = params.get("num_blocks", 4)
    filters = params.get("initial_filters", 32)

    for i in range(num_blocks):
        # Double filters every other block
        if i % 2 == 0 and i > 0:
            filters *= 2

        # Residual connection
        residual = x

        # Main path
        x = layers.Conv3D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)

        # Adjust residual if needed
        if residual.shape[-1] != filters:
            residual = layers.Conv3D(filters, 1)(residual)

        x = layers.Add()([x, residual])
        x = layers.Activation("relu")(x)

        # Pooling every other block
        if i % 2 == 1:
            x = layers.MaxPooling3D(2)(x)

    # Global pooling and dense layers
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(params.get("dense_units", 256), activation="relu")(x)
    x = layers.Dropout(params.get("dropout_rate", 0.3))(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=x)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params.get("learning_rate", 0.001)),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
