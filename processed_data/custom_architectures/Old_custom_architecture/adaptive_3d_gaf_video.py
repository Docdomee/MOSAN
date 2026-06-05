from tensorflow import keras


def build_model(input_shape, num_classes, params):
    """
    Adaptive 3D GAF Video CNN that handles both 4D (validation) and 5D (runtime) inputs
    """

    # Extract hyperparameters with defaults
    num_conv_layers = params.get("num_conv_layers", 4)
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 3)
    dense_units = params.get("dense_units", 128)
    dropout_rate = params.get("dropout_rate", 0.3)
    learning_rate = params.get("learning_rate", 0.001)

    # Input layer
    inputs = keras.layers.Input(shape=input_shape)

    # Determine if input is 4D (validation) or 5D (runtime)
    x = inputs

    if len(input_shape) == 3:  # 4D tensor: (batch, height, width, channels)
        # Validation mode - use 2D convolutions
        print("Using 2D mode for validation")

        for i in range(num_conv_layers):
            current_filters = filters * (2 ** (i // 2))

            x = keras.layers.Conv2D(
                filters=current_filters, kernel_size=kernel_size, activation="relu", padding="same"
            )(x)

            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.MaxPooling2D(pool_size=2)(x)
            x = keras.layers.Dropout(dropout_rate)(x)

        x = keras.layers.GlobalAveragePooling2D()(x)

    else:  # 5D tensor: (batch, segments, height, width, channels)
        # Runtime mode - use 3D convolutions
        print("Using 3D mode for runtime")

        for i in range(num_conv_layers):
            current_filters = filters * (2 ** (i // 2))

            # Use smaller temporal kernel for 3D to avoid negative dimensions
            temporal_kernel = max(1, min(kernel_size - 2, 3))  # Ensure at least 1
            spatial_kernel = kernel_size

            x = keras.layers.Conv3D(
                filters=current_filters,
                kernel_size=(temporal_kernel, spatial_kernel, spatial_kernel),
                activation="relu",
                padding="same",
            )(x)

            x = keras.layers.BatchNormalization()(x)

            # Use appropriate pooling - only pool spatial dimensions if temporal is small
            if input_shape[1] > 2:  # If we have enough temporal dimensions
                x = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
            else:
                x = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)

            x = keras.layers.Dropout(dropout_rate)(x)

        # Global pooling for 3D
        x = keras.layers.GlobalAveragePooling3D()(x)

    # Common dense layers for both modes
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Dense(dense_units // 2, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    # Build and compile model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
