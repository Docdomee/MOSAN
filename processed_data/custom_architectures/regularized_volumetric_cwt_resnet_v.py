import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, params: dict):
    """
    Improved Volumetric CWT Model.
    
    Fix: Added dimensionality checks to prevent MaxPooling3D from reducing 
    dimensions to zero or negative values (Dimensional Collapse).
    """
    
    # --- Hyperparameter Extraction ---
    filters = min(params.get("filters", 32), 64) 
    num_conv_layers = params.get("num_conv_layers", 4)
    # Ensure depth is within the 'sweet spot' (4-5)
    num_conv_layers = max(4, min(num_conv_layers, 5))
    
    dropout_rate = max(params.get("dropout_rate", 0.5), 0.5)
    dense_units = min(params.get("dense_units", 128), 256)
    weight_decay = params.get("weight_decay", 1e-4)
    
    # L2 Regularizer
    reg = keras.regularizers.l2(weight_decay)

    inputs = keras.Input(shape=input_shape)
    x = inputs

    # --- Volumetric Feature Extraction Stack ---
    # Motif: [Conv3D -> BN -> ReLU -> MaxPool3D (Conditional)] x N
    for i in range(num_conv_layers):
        x = layers.Conv3D(
            filters=filters, 
            kernel_size=(3, 3, 3), 
            padding='same', 
            kernel_regularizer=reg,
            activation=None # BN before activation
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # CRITICAL FIX: Only pool if the current dimensions allow it.
        # We check the shape of the tensor. If any dimension (D, H, W) is <= 1, 
        # we skip pooling to avoid the 'Negative dimension size' ValueError.
        # Note: Since we are in a functional API, we check the shape of the tensor x.
        current_shape = x.shape
        # current_shape is (Batch, Depth, Height, Width, Channels)
        if current_shape[1] is not None and current_shape[1] > 1 and \
           current_shape[2] is not None and current_shape[2] > 1 and \
           current_shape[3] is not None and current_shape[3] > 1:
            x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        else:
            # If dimensions are too small, we skip pooling for this layer 
            # to maintain the tensor's existence.
            pass

    # --- Global Feature Aggregation ---
    # GlobalAvgPool3D is critical to combat the 'Shift Gap' by removing 
    # spatial dependency and reducing over-parameterization.
    x = layers.GlobalAveragePooling3D()(x)

    # --- Classification Head ---
    x = layers.Dense(
        units=dense_units, 
        kernel_regularizer=reg, 
        activation=None
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model