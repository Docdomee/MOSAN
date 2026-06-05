def build_model(input_shape, num_classes, params: dict):
    from keras import layers
    from tensorflow import keras

    # Extract hyperparameters from params dict
    num_conv_layers = params.get("num_conv_layers", 4)
    kernel_size = params.get("kernel_size", 3)
    filters = params.get("filters", 64)
    dense_units = params.get("dense_units", 256)
    dropout_rate = params.get("dropout_rate", 0.3)
    learning_rate = params.get("learning_rate", 0.005)

    # Build model
    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # Convolutional blocks with residual connections
    for i in range(num_conv_layers):
        # Adjust filters progressively (doubling every two layers for scalability)
        current_filters = filters * (2 ** (i // 2))

        # Adaptive dropout: increase dropout for deeper layers to enhance stability
        adaptive_dropout = dropout_rate + 0.05 * (i / max(1, num_conv_layers - 1))
        adaptive_dropout = min(adaptive_dropout, 0.5)  # Cap at 0.5

        # Convolutional layer (adapted to 3D)
        conv = layers.Conv3D(
            current_filters, (kernel_size, kernel_size, kernel_size), activation="relu", padding="same"
        )
        model.add(conv)

        # Add residual connection for layers > 0
        if i > 0:
            # Use 1x1 conv to match dimensions if needed (adapted to 3D)
            if current_filters != filters * (2 ** ((i - 1) // 2)):
                residual = layers.Conv3D(current_filters, (1, 1, 1), padding="same")(model.layers[-2].output)
            else:
                residual = model.layers[-2].output
            model.add(layers.Add()([model.layers[-1].output, residual]))

        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling3D((2, 2, 2)))
        model.add(layers.Dropout(adaptive_dropout))

    # Flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout_rate))  # Standard dropout for dense
    model.add(layers.Dense(num_classes, activation="softmax"))

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
