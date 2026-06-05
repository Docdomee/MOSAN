from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes, params):
    """
    Hybrid 3D GAF Video model using TimeDistributed 2D convolutions
    instead of Conv3D to handle shape compatibility issues
    """
    # Extract hyperparameters
    num_conv_layers = params.get("num_conv_layers", 4)
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 3)
    dense_units = params.get("dense_units", 256)
    dropout_rate = params.get("dropout_rate", 0.3)
    learning_rate = params.get("learning_rate", 0.001)

    # Input layer - expects (batch, segments, height, width, channels)
    inputs = keras.Input(shape=input_shape)

    # TimeDistributed 2D convolutions for temporal processing
    x = inputs

    # First block: TimeDistributed 2D conv + pooling
    for i in range(min(2, num_conv_layers)):
        x = layers.TimeDistributed(layers.Conv2D(filters, kernel_size, activation="relu", padding="same"))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        filters = min(filters * 2, 128)  # Cap at 128

    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)

    # Middle blocks: TimeDistributed 2D conv
    for i in range(2, num_conv_layers):
        x = layers.TimeDistributed(layers.Conv2D(filters, kernel_size, activation="relu", padding="same"))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        if i % 2 == 1:  # Add pooling every other layer
            x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)

    # Temporal aggregation: Average over time dimension
    x = layers.GlobalAveragePooling3D()(x)

    # Dense layers
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(dense_units // 2, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Build and compile model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
