def build_model(input_shape, num_classes, params):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    import numpy as np

    def squeeze_and_excitation_1d(input_tensor, ratio=4):
        """Squeeze-and-Excitation block for 1D data."""
        filters = input_tensor.shape[-1]
        se_shape = (1, filters)

        # Global average pooling
        se = layers.GlobalAveragePooling1D()(input_tensor)
        se = layers.Reshape(se_shape)(se)
        se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
        se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)

        # Apply attention weights
        return layers.Multiply()([input_tensor, se])

    # --- Core Architecture Parameters ---
    num_conv_layers = 6
    filters = 48
    kernel_size = 7
    dropout_rate = params.get("dropout_rate", 0.7)
    weight_decay = params.get("weight_decay", 0.0025)
    learning_rate = params.get("learning_rate", 0.0001)
    dense_units = max(24, params.get("dense_units", 48)) # Ensure minimum dense units

    reg = regularizers.l2(weight_decay)

    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Optional: Gaussian Noise for Robustness
    if params.get("use_data_augmentation", False):
        x = layers.GaussianNoise(0.1)(x)

    # --- Convolutional Backbone with Residual and Attention ---
    for i in range(num_conv_layers):
        shortcut = x

        # Conv Block
        x = layers.Conv1D(filters, kernel_size, padding='same', kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Add Squeeze-and-Excitation Attention
        x = squeeze_and_excitation_1d(x)

        # Residual Connection
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, padding='same', use_bias=False)(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)

        # Downsample
        x = layers.MaxPooling1D(2)(x)

    # --- Global Pooling ---
    # Use GlobalMaxPooling for better feature discrimination
    x = layers.GlobalMaxPooling1D()(x)

    # --- Dense Processing ---
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)

    # --- Output ---
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation("softmax", dtype="float32")(x)

    # --- Model Compilation ---
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model