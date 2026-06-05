import numpy as np
import tensorflow as tf
from tensorflow import keras

def build_model(input_shape, num_classes, params):
    """
    Standard 2D CNN Architecture.
    Suitable for: 2D_SPECTROGRAM, 2D_GAF, 2D_IMAGE, 2D_CWT_SCALOGRAM.
    """
    # Defensive cast
    input_shape = tuple(input_shape)
    
    min_dim = min(input_shape[0], input_shape[1])
    max_layers = int(np.log2(min_dim)) if min_dim > 0 else 1
    requested_layers = params.get("num_conv_layers", 2)
    num_conv_layers = min(requested_layers, max_layers)

    if num_conv_layers < requested_layers:
        print(f"[Standard 2D CNN] Warning: Reduced layers from {requested_layers} to {num_conv_layers} due to input size.")

    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # --- Data Augmentation ---
    if params.get("use_data_augmentation"):
        x = keras.layers.RandomFlip("horizontal")(x)
        x = keras.layers.RandomRotation(0.1)(x)
        x = keras.layers.RandomZoom(0.1)(x)
    
    # L1/L2 Regularization
    l1 = 1e-4 if params.get("use_l1_regularization") else 0.0
    l2 = 1e-4 if params.get("use_l2_regularization") else 0.0
    reg = keras.regularizers.L1L2(l1=l1, l2=l2) if (l1 > 0 or l2 > 0) else None

    # Convolutional Blocks
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
        
        # Residual Connection
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
    
    keras.layers.Dense(num_classes)(x)`n    outputs = keras.layers.Activation("softmax", dtype="float32")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Optimizer
    lr = params.get("learning_rate", 1e-3)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
