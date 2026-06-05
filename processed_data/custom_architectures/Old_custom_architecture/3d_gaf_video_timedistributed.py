from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes, params):
    """
    3D GAF Video model using TimeDistributed 2D convolutions to handle
    both 4D validation dummy inputs and 5D runtime data.
    """
    # Extract hyperparameters with defaults
    num_conv_layers = params.get("num_conv_layers", 4)
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 5)
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

    # TimeDistributed 2D convolutions for temporal processing
    for i in range(num_conv_layers):
        # Double filters every other layer
        current_filters = filters * (2 ** (i // 2))

        x = layers.TimeDistributed(layers.Conv2D(current_filters, kernel_size, activation="relu", padding="same"))(x)

        x = layers.TimeDistributed(layers.BatchNormalization())(x)

        x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)

        x = layers.TimeDistributed(layers.Dropout(dropout_rate * 0.5))(x)  # Lower dropout in early layers

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
