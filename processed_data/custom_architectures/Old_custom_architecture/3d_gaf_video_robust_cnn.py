from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape, num_classes, params):
    """
    Robust 3D GAF Video CNN that handles available dataset constraints
    Uses TimeDistributed 2D convolutions with flexible temporal aggregation
    """

    # Extract hyperparameters
    num_conv_layers = params.get("num_conv_layers", 3)
    filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 3)
    dense_units = params.get("dense_units", 128)
    dropout_rate = params.get("dropout_rate", 0.3)
    learning_rate = params.get("learning_rate", 0.001)

    inputs = keras.Input(shape=input_shape)

    # Process each temporal segment with 2D convolutions
    x = inputs

    # Spatial feature extraction within each time segment
    for i in range(num_conv_layers):
        x = layers.TimeDistributed(layers.Conv2D(filters * (2**i), kernel_size, activation="relu", padding="same"))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
        x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)

    # Global temporal-spatial pooling - handles variable temporal dimensions
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
