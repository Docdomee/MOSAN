def build_model(input_shape, num_classes, params: dict):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape)

    # Extract parameters with defaults
    num_conv_layers = params.get("num_conv_layers", 4)
    filters_base = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 5)
    dropout_rate = params.get("dropout_rate", 0.32)
    dense_units = params.get("dense_units", 128)

    x = inputs

    # Progressive filter scaling with TimeDistributed 2D convolutions
    for i in range(num_conv_layers):
        current_filters = filters_base * (2 ** min(i, 2))  # Progressive scaling up to 4x
        x = layers.TimeDistributed(layers.Conv2D(current_filters, kernel_size, activation="relu", padding="same"))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
        x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)

    # Spatiotemporal feature extraction
    x = layers.Conv3D(filters_base * 2, (3, 3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(2)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Global pooling and classification
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    # Conservative learning rate as identified in historical data
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 0.002))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model
