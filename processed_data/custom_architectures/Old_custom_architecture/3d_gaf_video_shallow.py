from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes, params):
    """
    3D GAF Video model with shallow architecture to handle 24x24 spatial dimensions
    without causing dimension collapse during pooling operations.
    """
    # Extract hyperparameters with defaults
    num_conv_layers = params.get("num_conv_layers", 3)
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 3)
    dense_units = params.get("dense_units", 128)
    dropout_rate = params.get("dropout_rate", 0.3)
    learning_rate = params.get("learning_rate", 0.002)

    # Input layer - handles both 4D (dummy) and 5D (runtime) inputs
    inputs = keras.Input(shape=input_shape)

    # Handle 5D input shape: (batch, segments, height, width, channels)
    # If input is 4D (dummy validation), reshape to 5D
    if len(input_shape) == 3:  # 4D tensor (batch, h, w, c) from validation
        x = layers.Reshape((1, input_shape[0], input_shape[1], input_shape[2]))(inputs)
    else:  # 5D tensor (batch, segments, h, w, c) from runtime
        x = inputs

    # TimeDistributed 2D convolutions with conservative pooling
    for i in range(num_conv_layers):
        # Use progressive filter increase but more conservatively
        current_filters = filters * (2 ** min(i, 2))  # Cap at 4x filters max

        x = layers.TimeDistributed(layers.Conv2D(current_filters, kernel_size, activation="relu", padding="same"))(x)

        x = layers.TimeDistributed(layers.BatchNormalization())(x)

        # Only pool every other layer to preserve spatial dimensions
        if i % 2 == 0 and i < num_conv_layers - 1:  # Don't pool in last layer
            x = layers.TimeDistributed(layers.MaxPooling2D(2, padding="same"))(x)  # Use padding to preserve dimensions

        x = layers.TimeDistributed(layers.Dropout(dropout_rate * 0.5))(x)

    # Global temporal pooling followed by spatial pooling
    x = layers.GlobalAveragePooling3D()(x)

    # Dense layers
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(dense_units // 2, activation="relu")(x)
    x = layers.Dropout(dropout_rate * 0.7)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Build and compile model
    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model
