def build_model(input_shape, num_classes, params: dict):
    from keras import layers
    from tensorflow import keras

    # Extract parameters with defaults based on successful patterns
    num_conv_layers = params.get("num_conv_layers", 4)
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 5)
    dense_units = params.get("dense_units", 256)
    dropout_rate = params.get("dropout_rate", 0.35)
    learning_rate = params.get("learning_rate", 0.007)

    # Input layer for 2D spectrogram data
    inputs = layers.Input(shape=input_shape)

    # Initial convolution with larger receptive field
    x = layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), activation="relu", padding="same")(
        inputs
    )
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Progressive convolution blocks with increasing filters
    for i in range(num_conv_layers - 1):
        # Double filters every other layer up to 128 max
        current_filters = min(filters * (2 ** (i // 2)), 128)

        x = layers.Conv2D(
            filters=current_filters, kernel_size=(kernel_size, kernel_size), activation="relu", padding="same"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            filters=current_filters, kernel_size=(kernel_size, kernel_size), activation="relu", padding="same"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)

    # Global pooling to handle variable spatial dimensions
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers with progressive dropout
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate * 1.5)(x)

    x = layers.Dense(dense_units // 2, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Create and compile model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
