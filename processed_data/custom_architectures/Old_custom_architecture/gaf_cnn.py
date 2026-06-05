from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes, params):
    """
    Custom CNN architecture optimized for GAF (Gramian Angular Field) images
    GAF images capture temporal correlations as spatial patterns
    """

    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # First convolutional block with larger receptive field for GAF patterns
    model.add(
        layers.Conv2D(
            filters=params.get("filters", 32),
            kernel_size=params.get("kernel_size", 5),
            padding="same",
            activation="relu",
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2))

    # Additional convolutional layers based on parameter
    num_conv_layers = params.get("num_conv_layers", 3)
    for i in range(1, num_conv_layers):
        # Gradually increase filters while reducing spatial dimensions
        filters_multiplier = min(2**i, 4)  # Cap at 4x base filters
        model.add(
            layers.Conv2D(
                filters=params.get("filters", 32) * filters_multiplier,
                kernel_size=params.get("kernel_size", 5) - (i * 2) if i < 2 else 3,  # Reduce kernel size
                padding="same",
                activation="relu",
            )
        )
        model.add(layers.BatchNormalization())
        if i < num_conv_layers - 1:  # Don't pool before final conv layer
            model.add(layers.MaxPooling2D(pool_size=2))

    # Global average pooling instead of flattening to reduce parameters
    model.add(layers.GlobalAveragePooling2D())

    # Dense layers with dropout
    dense_units = params.get("dense_units", 128)
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(params.get("dropout_rate", 0.3)))

    # Output layer
    model.add(layers.Dense(num_classes, activation="softmax"))

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 0.001))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model
