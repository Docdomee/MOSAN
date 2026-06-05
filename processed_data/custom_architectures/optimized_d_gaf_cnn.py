def build_model(input_shape, num_classes, params):
    """
    Optimized 2D CNN for GAF inputs, leveraging motif-based design recommendations:
    - Depth: 4-6 conv layers
    - Filters: Start from 64 or 128
    - Kernel size: 3 or 5
    - BatchNorm + Dropout + Regularization
    - Residual connections supported optionally
    """

    import tensorflow as tf
    from tensorflow import keras
    import numpy as np

    def _get_regularizer(params):
        l2_value = params.get("l2_reg", 1e-4)
        return keras.regularizers.l2(l2_value)

    min_dim = min(input_shape[0], input_shape[1])
    max_layers = int(np.log2(min_dim)) if min_dim > 0 else 1
    requested_layers = params.get("num_conv_layers", 5)
    num_conv_layers = min(requested_layers, max_layers)

    if num_conv_layers < requested_layers:
        print(
            f"[Builder Warning] 2D_GAF_CNN: Requested {requested_layers} layers, "
            f"but data dimension only supports {num_conv_layers}. Adjusting automatically."
        )

    inputs = keras.Input(shape=input_shape)
    x = inputs

    # --- Data Augmentation ---
    if params.get("use_data_augmentation", False):
        x = keras.layers.RandomFlip("horizontal")(x)
        x = keras.layers.RandomRotation(0.1)(x)
        x = keras.layers.RandomZoom(0.1)(x)

    reg = _get_regularizer(params)
    use_residual = params.get("use_residual", False)
    base_filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 3)
    dropout_rate = params.get("dropout_rate", 0.4)
    dense_units = params.get("dense_units", 128)

    for i in range(num_conv_layers):
        num_filters = min(512, base_filters * (2 ** (i // 2)))

        shortcut = x

        x = keras.layers.Conv2D(
            num_filters,
            (kernel_size, kernel_size),
            padding="same",
            kernel_regularizer=reg
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

        if use_residual:
            if shortcut.shape[-1] != num_filters:
                shortcut = keras.layers.Conv2D(num_filters, (1, 1), padding="same")(shortcut)
            x = keras.layers.Add()([x, shortcut])

        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(
        learning_rate=params.get("learning_rate", 3e-4)
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model