from tensorflow import keras


def build_model(input_shape, num_classes, params):
    """
    TimeDistributed 2D CNN for 3D GAF Video data
    Processes each temporal segment with 2D convolutions
    """

    # Extract hyperparameters with defaults
    num_conv_layers = params.get("num_conv_layers", 4)
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 3)
    dense_units = params.get("dense_units", 128)
    dropout_rate = params.get("dropout_rate", 0.3)
    learning_rate = params.get("learning_rate", 0.001)

    # Input layer - expects (batch, segments, height, width, channels)
    inputs = keras.layers.Input(shape=input_shape)

    # TimeDistributed 2D convolutions for each segment
    x = inputs

    # First convolutional block
    for i in range(num_conv_layers):
        # Double filters every other layer
        current_filters = filters * (2 ** (i // 2))

        x = keras.layers.TimeDistributed(
            keras.layers.Conv2D(filters=current_filters, kernel_size=kernel_size, activation="relu", padding="same")
        )(x)

        x = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(x)

        x = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=2))(x)

        x = keras.layers.TimeDistributed(keras.layers.Dropout(dropout_rate))(x)

    # Global average pooling across spatial dimensions for each segment
    x = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D())(x)

    # Temporal aggregation - process the sequence of segment features
    x = keras.layers.LSTM(64, return_sequences=False)(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    # Dense layers for classification
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
