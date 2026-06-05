from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes, params):
    """
    Build a 2D CNN model for spectrogram classification
    """
    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # Add convolutional blocks
    num_conv_layers = params.get("num_conv_layers", 3)
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 3)

    for i in range(num_conv_layers):
        # Double filters every 2 layers
        current_filters = filters * (2 ** (i // 2))

        model.add(
            layers.Conv2D(
                filters=current_filters, kernel_size=(kernel_size, kernel_size), padding="same", activation="relu"
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        # Add dropout after pooling
        if i < num_conv_layers - 1:  # Don't add dropout to last conv layer
            dropout_rate = params.get("dropout_rate", 0.3)
            model.add(layers.Dropout(dropout_rate))

    # Flatten and add dense layers
    model.add(layers.Flatten())

    dense_units = params.get("dense_units", 128)
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.BatchNormalization())

    dropout_rate = params.get("dropout_rate", 0.3)
    model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(num_classes, activation="softmax"))

    # Compile model
    learning_rate = params.get("learning_rate", 0.001)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model
