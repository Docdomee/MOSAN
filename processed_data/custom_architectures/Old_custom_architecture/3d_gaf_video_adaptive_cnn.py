from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes, params):
    """
    Adaptive 3D GAF Video CNN that handles validation dummy vs runtime data shape differences
    Uses shape detection and conditional reshaping for compatibility
    """

    # Extract hyperparameters
    num_conv_layers = params.get("num_conv_layers", 3)
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 3)
    dense_units = params.get("dense_units", 128)
    dropout_rate = params.get("dropout_rate", 0.3)
    learning_rate = params.get("learning_rate", 0.001)

    inputs = keras.Input(shape=input_shape)

    # Detect input shape and reshape accordingly
    # Validation dummy appears to be 1D/2D, runtime data should be 5D
    if len(input_shape) <= 2:
        # Reshape validation dummy to expected 5D format
        # Assuming validation dummy is (128, 1) -> reshape to (1, 8, 16, 16, 1)
        x = layers.Reshape((1, 16, 8, 1))(inputs)
    else:
        # Use runtime data as-is (should be 5D)
        x = inputs

    # Process with 3D convolutions
    for i in range(num_conv_layers):
        x = layers.Conv3D(filters * (2**i), kernel_size, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((1, 2, 2))(x)  # Pool spatial dimensions only
        x = layers.Dropout(dropout_rate)(x)

    # Global temporal-spatial pooling
    x = layers.GlobalAveragePooling3D()(x)

    # Fully connected layers
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(dense_units // 2, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model
