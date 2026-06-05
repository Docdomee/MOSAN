def build_model(input_shape, num_classes, params: dict):
    from keras import layers
    from tensorflow import keras

    def residual_block(x, filters, kernel_size, dropout_rate, strides=1):
        shortcut = x
        x = layers.Conv3D(filters, (kernel_size, kernel_size, kernel_size), strides=strides, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv3D(filters, (kernel_size, kernel_size, kernel_size), strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        if strides != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv3D(filters, (1, 1, 1), strides=strides, padding="same")(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv3D(
        params.get("filters", 64),
        (params.get("kernel_size", 3), params.get("kernel_size", 3), params.get("kernel_size", 3)),
        padding="same",
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)

    num_conv_layers = params.get("num_conv_layers", 4)
    for i in range(num_conv_layers - 1):
        strides = 2 if i == num_conv_layers - 2 else 1
        x = residual_block(
            x,
            params.get("filters", 64),
            params.get("kernel_size", 3),
            params.get("dropout_rate", 0.3),
            strides=strides,
        )

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(params.get("dense_units", 256))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(params.get("dropout_rate", 0.3))(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params.get("learning_rate", 0.001)),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
