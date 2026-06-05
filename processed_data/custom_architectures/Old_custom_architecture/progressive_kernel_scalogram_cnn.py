def build_model(input_shape, num_classes, params: dict):
    from keras import layers
    from tensorflow import keras

    # Extract parameters from params dict, with defaults
    num_conv_layers = params.get("num_conv_layers", 4)
    initial_kernel_size = params.get("initial_kernel_size", 3)
    kernel_increment = params.get("kernel_increment", 1)  # How much to increase kernel size per layer
    filters = params.get("filters", 64)
    dense_units = params.get("dense_units", 256)
    dropout_rate = params.get("dropout_rate", 0.35)
    learning_rate = params.get("learning_rate", 0.007)

    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # Convolutional layers with progressive kernel sizes
    current_kernel_size = initial_kernel_size
    for i in range(num_conv_layers):
        model.add(
            layers.Conv2D(
                filters * (2**i), (current_kernel_size, current_kernel_size), activation="relu", padding="same"
            )
        )
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.BatchNormalization())
        current_kernel_size += kernel_increment  # Increase kernel size progressively

    # Flatten and dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
