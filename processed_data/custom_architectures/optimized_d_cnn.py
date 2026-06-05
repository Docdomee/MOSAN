import tensorflow as tf
from tensorflow import keras
import numpy as np

def build_model(input_shape, num_classes, params):
    """
    Optimized 1D CNN based on pattern analysis findings:
    - Low parameter count (<50K)
    - High dropout (0.4-0.5)
    - Small kernel sizes (3 or 9)
    - Very low learning rates (1e-5 to 1e-4)
    - Batch normalization and residual connections
    """
    
    # Architecture constraints
    max_layers = min(4, int(np.log2(input_shape[0]))) if input_shape[0] > 0 else 1
    requested_layers = params.get("num_conv_layers", 3)
    num_conv_layers = min(requested_layers, max_layers)
    
    if num_conv_layers < requested_layers:
        print(f"[Builder Warning] Requested {requested_layers} layers, but data dimension only supports {num_conv_layers}. Adjusting automatically.")

    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Data augmentation - Gaussian noise for 1D signals
    if params.get("use_data_augmentation", False):
        x = keras.layers.GaussianNoise(0.05)(x)  # Reduced noise level
    
    # Regularization
    reg = keras.regularizers.l2(params.get("l2_reg", 1e-4))
    dropout_rate = params.get("dropout_rate", 0.45)  # High dropout as per findings
    
    # Initial convolution block
    x = keras.layers.Conv1D(
        filters=params.get("initial_filters", 16),
        kernel_size=params.get("kernel_size", 3),
        padding="same",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling1D(2)(x)
    
    # Main convolutional blocks with residual connections
    for i in range(1, num_conv_layers):
        num_filters = min(128, params.get("initial_filters", 16) * (2 ** i))  # Cap at 128 filters
        
        # Residual connection
        shortcut = x
        
        # First conv layer
        x = keras.layers.Conv1D(
            num_filters,
            params.get("kernel_size", 3),
            padding="same",
            kernel_regularizer=reg
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Second conv layer
        x = keras.layers.Conv1D(
            num_filters,
            params.get("kernel_size", 3),
            padding="same",
            kernel_regularizer=reg
        )(x)
        x = keras.layers.BatchNormalization()(x)
        
        # Adjust shortcut for residual connection
        if shortcut.shape[-1] != num_filters:
            shortcut = keras.layers.Conv1D(num_filters, 1, padding="same")(shortcut)
        
        # Add residual connection
        x = keras.layers.Add()([x, shortcut])
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling1D(2)(x)
    
    # Global pooling and classification head
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers with dropout
    x = keras.layers.Dense(
        params.get("dense_units", 64),
        activation="relu",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    
    # Model compilation with low learning rate
    model = keras.Model(inputs, outputs)
    
    # Use very low learning rate as per findings
    learning_rate = params.get("learning_rate", 5e-5)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model