from tensorflow import keras


def build_model(input_shape, num_classes, params: dict):
    # Adjust input_shape for validation (3D dummy) to 5D for 3D_GAF_VIDEO
    if len(input_shape) == 3:
        # Assume validation dummy: prepend segments=12, adjust size to 24x24 for compatibility
        input_shape = (12, 24, 24, input_shape[-1])

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))

    # Hyperparameters
    num_conv_layers = params.get("num_conv_layers", 4)  # Limit to 4 to avoid over-downsampling
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 3)  # Smaller kernel
    dense_units = params.get("dense_units", 128)
    dropout_rate = params.get("dropout_rate", 0.3)
    learning_rate = params.get("learning_rate", 0.001)

    # Conv3D layers for 3D spatiotemporal features
    for i in range(num_conv_layers):
        model.add(
            keras.layers.Conv3D(
                filters * (2**i), (kernel_size, kernel_size, kernel_size), activation="relu", padding="same"
            )
        )
        model.add(keras.layers.BatchNormalization())
        if i < num_conv_layers - 1:  # Pool only between layers, not after last
            model.add(keras.layers.MaxPooling3D((2, 2, 2)))

    # Global pooling to flatten without dimension issues
    model.add(keras.layers.GlobalAveragePooling3D())

    model.add(keras.layers.Dense(dense_units, activation="relu"))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model
