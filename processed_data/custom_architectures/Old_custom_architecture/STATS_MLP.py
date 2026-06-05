from tensorflow import keras


def build_model(input_shape, num_classes, params: dict):
    # STATS_MLP: Simple MLP for statistical features (tabular data)
    # Input shape: (num_features,)
    model = keras.Sequential()

    # First dense layer
    model.add(keras.layers.Dense(params.get("dense_units", 128), activation="relu", input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(params.get("dropout_rate", 0.3)))

    # Second dense layer (optional, based on params)
    if params.get("num_dense_layers", 1) > 1:
        model.add(keras.layers.Dense(params.get("dense_units", 128), activation="relu"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(params.get("dropout_rate", 0.3)))

    # Output layer
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    # Compile
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 0.001))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return model
