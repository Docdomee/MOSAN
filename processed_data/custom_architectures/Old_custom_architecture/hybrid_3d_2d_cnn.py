def build_model(input_shape, num_classes, params: dict):
    from keras import layers
    from tensorflow import keras

    # Extract parameters with defaults
    num_conv_layers = params.get("num_conv_layers", 4)
    filters_base = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 7)
    dense_units = params.get("dense_units", 256)
    dropout_rate = params.get("dropout_rate", 0.32)
    learning_rate = params.get("learning_rate", 0.002)

    inputs = layers.Input(shape=input_shape)

    # TimeDistributed 2D convolutions for spatial feature extraction
    x = inputs
    for i in range(num_conv_layers):
        filters = filters_base * (2**i)
        x = layers.TimeDistributed(layers.Conv2D(filters, kernel_size, padding="same", activation="relu"))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
        x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)

    # Temporal convolution to capture time dependencies
    x = layers.Conv3D(filters_base * 2, (3, 1, 1), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D((2, 1, 1))(x)

    # Global pooling and dense layers
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units // 2, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
