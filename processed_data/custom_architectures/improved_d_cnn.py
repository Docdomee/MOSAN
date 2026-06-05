def build_model(input_shape, num_classes, params):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    # --- Helper Function ---
    def _get_regularizer(params):
        reg_type = params.get("regularization_type", "l2")
        reg_rate = params.get("regularization_rate", 1e-4)
        if reg_type == "l1":
            return keras.regularizers.l1(reg_rate)
        elif reg_type == "l2":
            return keras.regularizers.l2(reg_rate)
        elif reg_type == "l1_l2":
            return keras.regularizers.l1_l2(l1=reg_rate, l2=reg_rate)
        else:
            return None

    # --- Model Configuration ---
    max_layers = int(np.log2(input_shape[0])) if input_shape[0] > 1 else 1
    requested_layers = params.get("num_conv_layers", 2)
    num_conv_layers = min(requested_layers, max_layers)

    if num_conv_layers < requested_layers:
        print(
            f"[Builder Warning] Requested {requested_layers} layers, "
            f"but data dimension only supports {num_conv_layers}. Adjusting automatically."
        )

    inputs = keras.Input(shape=input_shape)
    x = inputs

    # --- Data Augmentation ---
    if params.get("use_data_augmentation", False):
        x = layers.GaussianNoise(stddev=params.get("aug_noise_stddev", 0.05))(x)

    reg = _get_regularizer(params)

    # --- Convolutional Layers ---
    for i in range(num_conv_layers):
        num_filters = min(512, params.get("base_filters", 32) * (2 ** i))
        kernel_size = params.get("kernel_size", 3)
        use_residual = params.get("use_residual", True)

        # Residual connection setup
        shortcut = x

        # First conv layer
        x = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=reg,
            kernel_initializer="he_normal"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # Second conv layer (for potential residual)
        x = layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=reg,
            kernel_initializer="he_normal"
        )(x)

        # Residual connection
        if use_residual:
            if shortcut.shape[-1] != num_filters:
                shortcut = layers.Conv1D(
                    filters=num_filters,
                    kernel_size=1,
                    padding="same",
                    kernel_regularizer=reg
                )(shortcut)
            x = layers.Add()([x, shortcut])

        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(params.get("conv_dropout_rate", 0.1))(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

    # --- Head ---
    x = layers.GlobalAveragePooling1D()(x)

    # Dense layers
    for _ in range(params.get("num_dense_layers", 1)):
        x = layers.Dense(
            units=params.get("dense_units", 128),
            activation="relu",
            kernel_regularizer=reg
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(params.get("dense_dropout_rate", 0.5))(x)

    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    # --- Compile ---
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model