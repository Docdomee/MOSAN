import numpy as np
import tensorflow as tf
from tensorflow import keras

def _get_regularizer(params):
    l1 = 1e-4 if params.get("use_l1_regularization") else 0.0
    l2 = 1e-4 if params.get("use_l2_regularization") else 0.0
    if l1 > 0 or l2 > 0:
        return keras.regularizers.L1L2(l1=l1, l2=l2)
    return None

def build_1d_cnn(input_shape, num_classes, params):
    # Calcola il numero massimo di layer possibili in base alla dimensione dell'input
    max_layers = int(np.log2(input_shape[0])) if input_shape[0] > 0 else 1
    requested_layers = params.get("num_conv_layers", 2)
    # Usa il valore più piccolo tra quello richiesto e quello massimo possibile per evitare crash
    num_conv_layers = min(requested_layers, max_layers)

    if num_conv_layers < requested_layers:
        print(
            f"[Builder Warning] 1D_CNN: Requested {requested_layers} layers, but data dimension only supports {num_conv_layers}. Adjusting automatically."
        )

    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # --- Data Augmentation (1D) ---
    if params.get("use_data_augmentation"):
        # For 1D signals, standard image augmentation (flip/rotate) is often mathematically invalid or specific.
        # We use simple implementation safe for generic signals: Gaussian Noise and simple Shift
        # Note: Keras doesn't have native 1D augmentation layers as standard as 2D.
        # We'll stick to a simple GaussianNoise for robustness if requested.
        x = keras.layers.GaussianNoise(0.1)(x)
    # ------------------------------

    reg = _get_regularizer(params)

    for i in range(num_conv_layers):
        num_filters = min(512, params.get("filters", 32) * (2**i))
        
        # Residual Connection
        shortcut = x
        
        x = keras.layers.Conv1D(
            num_filters, 
            params.get("kernel_size", 3), 
            padding="same",
            kernel_regularizer=reg
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        if params.get("use_residual", False):
            if shortcut.shape[-1] != num_filters:
                shortcut = keras.layers.Conv1D(num_filters, 1, padding="same")(shortcut)
            x = keras.layers.Add()([x, shortcut])
            
        x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(
        params.get("dense_units", 128), 
        activation="relu",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Dropout(params.get("dropout_rate", 0.5))(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_2d_spectrogram_cnn(input_shape, num_classes, params):
    min_dim = min(input_shape[0], input_shape[1])
    max_layers = int(np.log2(min_dim)) if min_dim > 0 else 1
    requested_layers = params.get("num_conv_layers", 2)
    num_conv_layers = min(requested_layers, max_layers)

    if num_conv_layers < requested_layers:
        print(
            f"[Builder Warning] 2D_CNN: Requested {requested_layers} layers, but data dimension only supports {num_conv_layers}. Adjusting automatically."
        )

    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # --- Data Augmentation (2D) ---
    if params.get("use_data_augmentation"):
        x = keras.layers.RandomFlip("horizontal")(x)
        x = keras.layers.RandomRotation(0.1)(x)
        x = keras.layers.RandomZoom(0.1)(x)
    # ------------------------------
    
    reg = _get_regularizer(params)

    for i in range(num_conv_layers):
        num_filters = min(512, params.get("filters", 32) * (2**i))
        
        shortcut = x
        
        x = keras.layers.Conv2D(
            num_filters, 
            (params.get("kernel_size", 3), params.get("kernel_size", 3)), 
            padding="same",
            kernel_regularizer=reg
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        if params.get("use_residual", False):
            if shortcut.shape[-1] != num_filters:
                shortcut = keras.layers.Conv2D(num_filters, (1, 1), padding="same")(shortcut)
            x = keras.layers.Add()([x, shortcut])
            
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(
        params.get("dense_units", 128), 
        activation="relu",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Dropout(params.get("dropout_rate", 0.5))(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_3d_video_cnn(input_shape, num_classes, params):
    min_dim = min(input_shape[1], input_shape[2])
    max_layers = int(np.log2(min_dim)) if min_dim > 0 else 1
    requested_layers = params.get("num_conv_layers", 2)
    num_conv_layers = min(requested_layers, max_layers)

    if num_conv_layers < requested_layers:
        print(
            f"[Builder Warning] 3D_CNN: Requested {requested_layers} layers, but data dimension only supports {num_conv_layers}. Adjusting automatically."
        )

    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    reg = _get_regularizer(params)

    for i in range(num_conv_layers):
        num_filters = min(512, params.get("filters", 32) * (2**i))
        
        shortcut = x
        
        x = keras.layers.Conv3D(num_filters, kernel_size=params.get("kernel_size", 3), padding="same", kernel_regularizer=reg)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        if params.get("use_residual", False):
            if shortcut.shape[-1] != num_filters:
                shortcut = keras.layers.Conv3D(num_filters, kernel_size=1, padding="same")(shortcut)
            x = keras.layers.Add()([x, shortcut])
            
        x = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    x = keras.layers.GlobalAveragePooling3D()(x)
    x = keras.layers.Dense(params.get("dense_units", 128), activation="relu", kernel_regularizer=reg)(x)
    x = keras.layers.Dropout(params.get("dropout_rate", 0.5))(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


MODEL_BUILDERS = {
    "1D_CNN": build_1d_cnn,
    
    # STANDARD IMAGE KEY (Primary)
    "2D_IMAGE": build_2d_spectrogram_cnn,
    
    "2D_SPECTROGRAM": build_2d_spectrogram_cnn,
    "2D_GAF": build_2d_spectrogram_cnn,
    "2D_CWT_SCALOGRAM": build_2d_spectrogram_cnn,
    
    # LEGACY / ALIAS KEYS (Do not remove)
    "2D_GENERIC_IMAGE": build_2d_spectrogram_cnn, # Alias for backward compatibility
    
    # 3D VIDEO MODELS
    "3D_VIDEO": build_3d_video_cnn,
    "3D_GAF_VIDEO": build_3d_video_cnn,
    "3D_DYNAMIC_GAF": build_3d_video_cnn,
    "3D_DYNAMIC_CWT": build_3d_video_cnn,
    "3D_WAVELET_CWT": build_3d_video_cnn,
}