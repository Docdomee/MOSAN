import numpy as np
import tensorflow as tf
from tensorflow import keras

def build_model(input_shape, num_classes, params):
    """
    Residual 2D CNN for GAF data.
    
    This model implements a residual CNN architecture specifically designed for 2D_GAF data.
    It builds upon the base 2D CNN with residual connections, inspired by the parameters
    observed in the 1D CNN evolution. The model uses 2D convolutions with batch normalization
    and ReLU activation. Residual connections are added to help with gradient flow in deeper
    networks. The architecture adapts insights from the 1D CNN parameter evolution, such as
    potential residual patterns and regularization strategies, to the 2D domain.
    
    Key improvements:
    - Residual connections for better gradient flow
    - Configurable number of layers and filters
    - Data augmentation for 2D GAF images
    - Global average pooling to reduce parameters
    - Configurable regularization and dropout
    """
    
    # Helper function to get regularizer
    def _get_regularizer(params):
        reg_type = params.get("regularizer", None)
        if reg_type == "l1":
            return keras.regularizers.L1(l1=params.get("l1_reg", 1e-4))
        elif reg_type == "l2":
            return keras.regularizers.L2(l2=params.get("l2_reg", 1e-4))
        elif reg_type == "l1_l2":
            return keras.regularizers.L1L2(l1=params.get("l1_reg", 1e-4), l2=params.get("l2_reg", 1e-4))
        else:
            return None

    # Calculate maximum allowable layers based on input dimensions
    min_dim = min(input_shape[0], input_shape[1])
    max_layers = int(np.log2(min_dim)) if min_dim > 0 else 1
    requested_layers = params.get("num_conv_layers", 2)
    num_conv_layers = min(requested_layers, max_layers)

    if num_conv_layers < requested_layers:
        print(
            f"[Builder Warning] 2D_CNN: Requested {requested_layers} layers, but data dimension only supports {num_conv_layers}. Adjusting automatically."
        )

    # Model input
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Data Augmentation for 2D GAF data
    if params.get("use_data_augmentation", False):
        x = keras.layers.RandomFlip("horizontal")(x)
        x = keras.layers.RandomRotation(0.1)(x)
        x = keras.layers.RandomZoom(0.1)(x)
    
    # Regularizer
    reg = _get_regularizer(params)

    # Initial convolution layer
    x = keras.layers.Conv2D(
        params.get("initial_filters", 32), 
        (params.get("kernel_size", 3), params.get("kernel_size", 3)), 
        padding="same",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Residual blocks
    for i in range(num_conv_layers):
        num_filters = min(512, params.get("filters", 32) * (2**i))
        
        # First conv layer in the block
        shortcut = x
        x = keras.layers.Conv2D(
            num_filters, 
            (params.get("kernel_size", 3), params.get("kernel_size", 3)), 
            padding="same",
            kernel_regularizer=reg
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Second conv layer in the block
        x = keras.layers.Conv2D(
            num_filters, 
            (params.get("kernel_size", 3), params.get("kernel_size", 3)), 
            padding="same",
            kernel_regularizer=reg
        )(x)
        x = keras.layers.BatchNormalization()(x)
        
        # Adjust shortcut for residual connection if needed
        if shortcut.shape[-1] != num_filters:
            shortcut = keras.layers.Conv2D(num_filters, (1, 1), padding="same")(shortcut)
        
        # Add residual connection
        x = keras.layers.Add()([x, shortcut])
        x = keras.layers.Activation("relu")(x)
        
        # Pooling layer
        x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Global average pooling and classifier
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(
        params.get("dense_units", 128), 
        activation="relu",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Dropout(params.get("dropout_rate", 0.5))(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    # Create model
    model = keras.Model(inputs, outputs)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model