import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def build_model(input_shape, num_classes, params):
    """
    Deep Residual 2D-CNN Architecture for GAF Images from SERS Spectra.
    
    This model adapts the successful patterns from 1D-CNN trials but restructures
    for 2D GAF image input. It uses residual connections, moderate kernel sizes,
    and progressive depth to optimize performance on GAF representations.
    
    Key Design Elements:
    - Residual blocks with projection shortcuts for stable deep training
    - Conservative kernel sizes (3x3) throughout
    - Progressive filter expansion (32 -> 64 -> 128 -> 256)
    - Spatial dropout for 2D regularization
    - Global average pooling to preserve spatial information
    """

    # Defensive cast
    input_shape = tuple(input_shape)
    
    # Parse hyperparameters with defaults aligned to Pattern Scout findings
    dropout_rate = params.get("dropout_rate", 0.54)
    l2_reg = params.get("l2_reg", 1e-4)
    use_residual = params.get("use_residual", True)
    filters_list = params.get("filters_list", [32, 64, 128, 256])
    blocks_per_stage = params.get("blocks_per_stage", 3)
    learning_rate = params.get("learning_rate", 1e-4)
    
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Initial Conv Layer
    x = layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Residual Stages
    for stage, filters in enumerate(filters_list):
        for block in range(blocks_per_stage):
            # Apply stride=2 for downsampling in first block of each stage after the first
            stride = 2 if (stage > 0 and block == 0) else 1
            shortcut = x

            # First conv layer in block
            y = layers.Conv2D(
                filters,
                kernel_size=3,
                strides=stride,
                padding="same",
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                kernel_initializer="he_normal"
            )(x)
            y = layers.BatchNormalization()(y)
            y = layers.Activation("relu")(y)
            
            # Second conv layer in block
            y = layers.Conv2D(
                filters,
                kernel_size=3,
                padding="same",
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                kernel_initializer="he_normal"
            )(y)
            y = layers.BatchNormalization()(y)

            # Projection shortcut if needed
            if stride != 1 or shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(
                    filters,
                    kernel_size=1,
                    strides=stride,
                    padding="same",
                    kernel_regularizer=keras.regularizers.l2(l2_reg)
                )(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)

            # Residual connection
            if use_residual:
                y = layers.Add()([shortcut, y])
            
            x = layers.Activation("relu")(y)
            
        # Dropout after each stage to control overfitting
        x = layers.SpatialDropout2D(dropout_rate)(x)

    # Global Average Pooling instead of Flatten + Dense
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense bottleneck with dropout
    x = layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)

    # Build model and compile
    model = keras.Model(inputs, outputs, name="DeepResGAF2D")
    
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        epsilon=1e-8
    )
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model