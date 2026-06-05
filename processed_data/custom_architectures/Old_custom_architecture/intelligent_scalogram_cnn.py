def build_model(input_shape, num_classes, params: dict):
    from keras import layers
    from tensorflow import keras

    # Extract parameters with defaults
    num_conv_layers = params.get("num_conv_layers", 7)
    kernel_size = params.get("kernel_size", 9)
    initial_filters = params.get("initial_filters", 32)
    dense_units = params.get("dense_units", 128)
    dropout_rate = params.get("dropout_rate", 0.35)
    learning_rate = params.get("learning_rate", 0.0003)

    inputs = keras.Input(shape=input_shape)

    x = inputs
    filters = initial_filters

    # Convolutional layers with progressive filter doubling and batch norm
    for i in range(num_conv_layers):
        x = layers.Conv2D(filters, (kernel_size, kernel_size), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        if i < num_conv_layers - 1:  # Max pool except last
            x = layers.MaxPooling2D((2, 2))(x)
        filters *= 2  # Double filters progressively

    # Global average pooling to reduce dimensions
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model
