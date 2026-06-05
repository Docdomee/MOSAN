import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_model(input_shape, num_classes, params):
    """
    Improved ResNet-based 2D GAF Model.
    
    This model enhances the standard 2D_GAF approach by implementing a ResNet-inspired
    architecture with residual connections, batch normalization, and adaptive depth.
    It follows the design strategy derived from the initial 1D_CNN node vector while
    adapting to 2D GAF inputs.
    """
    
    # --- Helper Functions ---
    def _get_regularizer(params):
        reg_type = params.get("regularizer", "l2")
        reg_rate = params.get("regularizer_rate", 1e-4)
        if reg_type == "l1":
            return keras.regularizers.l1(reg_rate)
        elif reg_type == "l2":
            return keras.regularizers.l2(reg_rate)
        elif reg_type == "l1_l2":
            return keras.regularizers.l1_l2(l1=reg_rate, l2=reg_rate)
        else:
            return None

    def residual_block(x, filters, stride=1, conv_shortcut=False, reg=None):
        """Standard ResNet residual block with two convolutions."""
        shortcut = x

        x = layers.Conv2D(filters, 3, strides=stride, padding='same', kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters, 3, strides=1, padding='same', kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)

        if conv_shortcut or stride != 1:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', kernel_regularizer=reg)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        return x

    # --- Model Construction ---
    reg = _get_regularizer(params)
    
    # Handle input shape dynamically
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Initial conv layer
    x = layers.Conv2D(
        params.get("initial_filters", 32), 
        (params.get("initial_kernel_size", 7), params.get("initial_kernel_size", 7)),
        strides=(2, 2),
        padding='same',
        kernel_regularizer=reg
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # ResNet blocks
    # Stage 1
    x = residual_block(x, 32, reg=reg)
    x = residual_block(x, 32, reg=reg)
    
    # Stage 2
    x = residual_block(x, 64, stride=2, conv_shortcut=True, reg=reg)
    x = residual_block(x, 64, reg=reg)
    
    # Stage 3
    x = residual_block(x, 128, stride=2, conv_shortcut=True, reg=reg)
    x = residual_block(x, 128, reg=reg)
    
    # Stage 4
    x = residual_block(x, 256, stride=2, conv_shortcut=True, reg=reg)
    x = residual_block(x, 256, reg=reg)

    # Global pooling and classifier
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with regularization
    if params.get("use_dense", True):
        x = layers.Dense(
            params.get("dense_units", 128),
            activation='relu',
            kernel_regularizer=reg
        )(x)
        x = layers.Dropout(params.get("dropout_rate", 0.5))(x)
    
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = keras.Model(inputs, outputs)
    
    # Compile with optimizer
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model