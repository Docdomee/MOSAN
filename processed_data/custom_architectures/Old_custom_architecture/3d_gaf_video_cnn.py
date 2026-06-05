def build_model(input_shape, num_classes, params: dict):
    from tensorflow import keras

    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Handle validation dummy (3D shape, leading to 4D input) by reshaping to 5D
    if len(input_shape) == 3:
        height, width, channels = input_shape
        x = keras.layers.Reshape((1, height, width, channels))(x)

    # Now assume 5D input (segments, height, width, channels)
    num_conv_layers = params.get("num_conv_layers", 3)
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 3)

    for i in range(num_conv_layers):
        x = keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)

        # Conditional pooling: only pool spatial dimensions if they're > 1
        # Use strides (1,2,2) to preserve temporal dimension
        if i < num_conv_layers - 1:
            x = keras.layers.MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(x)
        else:
            # For the last layer, use global pooling to avoid negative output sizes
            x = keras.layers.GlobalMaxPooling3D()(x)

    # If not using global pooling in last layer, flatten
    if num_conv_layers == 0 or not isinstance(x, keras.layers.GlobalMaxPooling3D):
        x = keras.layers.Flatten()(x)

    dense_units = params.get("dense_units", 256)
    dropout_rate = params.get("dropout_rate", 0.3)
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(dense_units // 2, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    learning_rate = params.get("learning_rate", 0.001)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
