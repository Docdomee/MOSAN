import tensorflow as tf
from tensorflow import keras
import numpy as np

def build_model(input_shape, num_classes, params):
    """
    Advanced 1D CNN with Multi-Scale Kernels and Enhanced Regularization
    
    Implements the design strategy based on Pattern Scout analysis:
    - Alternating kernel sizes (3-5-7) for multi-scale feature extraction
    - High dropout rates (>0.5) for better generalization
    - Controlled capacity with 256 dense units
    - Proper regularization and learning rate scheduling
    """
    
    # --- Data Augmentation ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    if params.get("use_data_augmentation", False):
        x = keras.layers.GaussianNoise(0.1)(x)
    
    # --- Regularization ---
    reg = keras.regularizers.l2(params.get("weight_decay", 1e-4))
    
    # --- Convolutional Layers ---
    # Fixed to 8 layers as per design strategy
    num_conv_layers = 8
    base_filters = params.get("base_filters", 128)
    
    # Alternating kernel sizes pattern: [3, 5, 7, 3, 5, 7, 3, 5]
    kernel_sizes = [3, 5, 7, 3, 5, 7, 3, 5]
    
    for i in range(num_conv_layers):
        # Gradually increase filters but keep within reasonable bounds
        num_filters = min(base_filters * (2 ** (i // 3)), 256)
        kernel_size = kernel_sizes[i]
        
        # Residual connection
        shortcut = x
        
        # Convolutional block
        x = keras.layers.Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=reg,
            kernel_initializer="he_normal"
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Add residual connection
        if shortcut.shape[-1] != num_filters:
            shortcut = keras.layers.Conv1D(
                num_filters, 1, padding="same", kernel_regularizer=reg
            )(shortcut)
        x = keras.layers.Add()([x, shortcut])
        
        # Max pooling after every 2 layers to control dimensionality
        if (i + 1) % 2 == 0:
            x = keras.layers.MaxPooling1D(2)(x)
        
        # Add dropout after every 2-3 layers with high rate (>0.5)
        if (i + 1) % 3 == 0 or (i + 1) % 2 == 0:
            dropout_rate = params.get("dropout_rate", 0.55)
            # Ensure dropout is within safe range
            dropout_rate = max(0.5, min(0.6, dropout_rate))
            x = keras.layers.Dropout(dropout_rate)(x)
    
    # --- Global Average Pooling ---
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # --- Dense Layers ---
    # Limited to 256 units as per design strategy
    dense_units = min(params.get("dense_units", 256), 256)
    x = keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=reg
    )(x)
    
    # High dropout before final classification
    final_dropout = params.get("final_dropout_rate", 0.55)
    final_dropout = max(0.5, min(0.6, final_dropout))
    x = keras.layers.Dropout(final_dropout)(x)
    
    # --- Output Layer ---
    outputs = keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32"
    )(x)
    
    # --- Model Compilation ---
    model = keras.Model(inputs, outputs)
    
    # Use lower learning rate as per design strategy
    learning_rate = params.get("learning_rate", 5e-4)
    # Ensure learning rate is within safe range
    learning_rate = max(1e-5, min(1e-2, learning_rate))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model