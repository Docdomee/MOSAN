def build_model(input_shape, num_classes, params: dict):
    from tensorflow import keras
    from tensorflow.keras import layers

    # Default hyperparameters
    num_conv_layers = params.get("num_conv_layers", 4)
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 5)
    dense_units = params.get("dense_units", 256)
    dropout_rate = params.get("dropout_rate", 0.32)
    learning_rate = params.get("learning_rate", 0.005)

    model = keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    for i in range(num_conv_layers):
        model.add(layers.Conv2D(filters, (kernel_size, kernel_size), activation="relu", padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
