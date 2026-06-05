def build_model(input_shape, num_classes, params: dict):
    import tensorflow as tf
    from keras import layers
    from tensorflow import keras

    # Extract parameters with defaults
    num_conv_layers = params.get("num_conv_layers", 4)
    kernel_size = params.get("kernel_size", 3)
    filters = params.get("filters", 64)
    dense_units = params.get("dense_units", 256)
    dropout_rate = params.get("dropout_rate", 0.3)
    learning_rate = params.get("learning_rate", 0.005)

    # Model definition
    inputs = keras.Input(shape=input_shape)

    # Handle input shape: expand dims if 3D (for validation compatibility)
    if len(input_shape) == 3:
        x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)  # Add temporal dim: (batch, 1, h, w, c)
    else:
        x = inputs  # Assume 4D (segments, h, w, c) for real data

    # Convolutional blocks with progressive filters for 3D spatiotemporal data
    for i in range(num_conv_layers):
        x = layers.Conv3D(filters * (2**i), (1, kernel_size, kernel_size), activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((1, 2, 2))(x)  # Pool spatial dimensions, keep temporal

    # Global pooling and dense layers
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
