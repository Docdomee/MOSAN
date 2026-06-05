from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes, params):
    """
    Optimized 3D CNN architecture for GAF video data
    Handles both 4D validation dummy inputs and 5D runtime data
    """

    # Extract hyperparameters with defaults
    num_conv_layers = params.get("num_conv_layers", 4)
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 5)
    dense_units = params.get("dense_units", 128)
    dropout_rate = params.get("dropout_rate", 0.3)
    learning_rate = params.get("learning_rate", 0.002)

    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Handle 4D vs 5D input compatibility
    if len(input_shape) == 3:  # 4D input (batch, height, width, channels)
        # This is likely validation dummy data - reshape to 5D
        # Add temporal dimension of size 1 for compatibility
        x = layers.Reshape((1,) + input_shape)(x)
    # else: already 5D (batch, segments, height, width, channels)

    # 3D convolutional blocks with batch normalization
    for i in range(num_conv_layers):
        # Double filters every other layer
        current_filters = filters * (2 ** (i // 2))

        x = layers.Conv3D(
            filters=current_filters,
            kernel_size=(3, kernel_size, kernel_size),  # (temporal, spatial, spatial)
            padding="same",
            activation="relu",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)  # Only pool spatial dimensions
        x = layers.Dropout(dropout_rate)(x)

    # Global average pooling to handle variable spatial dimensions
    x = layers.GlobalAveragePooling3D()(x)

    # Dense layers
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model
