from tensorflow import keras


def build_model(input_shape, num_classes, params):
    """
    2D CNN architecture for spectrogram data
    """
    model = keras.Sequential()

    # Input layer
    model.add(keras.layers.Input(shape=input_shape))

    # Convolutional layers
    num_conv_layers = params.get("num_conv_layers", 3)
    filters = params.get("filters", 32)
    kernel_size = params.get("kernel_size", 3)

    for i in range(num_conv_layers):
        model.add(
            keras.layers.Conv2D(
                filters=filters * (2**i),  # Increase filters with depth
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )
        )
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=2))

    # Flatten and dense layers
    model.add(keras.layers.Flatten())

    dropout_rate = params.get("dropout_rate", 0.3)
    model.add(keras.layers.Dropout(dropout_rate))

    dense_units = params.get("dense_units", 128)
    model.add(keras.layers.Dense(dense_units, activation="relu"))
    model.add(keras.layers.BatchNormalization())

    # Output layer
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    # Compile model
    learning_rate = params.get("learning_rate", 0.001)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model
