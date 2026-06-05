import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_model(input_shape, num_classes, params):
    """
    Adaptive Improved ResNet Architecture.
    Automatically detects input dimensionality (1D, 2D, or 3D) and selects appropriate layers.
    Respects 'num_conv_layers' for depth optimization.
    """
    
    # --- Parameters ---
    dropout_rate = params.get("dropout_rate", 0.25)
    learning_rate = params.get("learning_rate", 1e-3)
    reg_strength = params.get("l2_reg", 1e-4)
    num_conv_layers = params.get("num_conv_layers", 10) # Target depth from sweep
    reg = keras.regularizers.l2(reg_strength)
    
    # --- Detect Dimensionality ---
    # spatial_dims: 1 for 1D, 2 for 2D, 3 for 3D (Time + 2D)
    spatial_dims = len(input_shape) - 1
    
    if spatial_dims == 1:
        Conv, Pool, GAP = layers.Conv1D, layers.MaxPooling1D, layers.GlobalAveragePooling1D
        k_init, k_res = 5, 3
    elif spatial_dims == 2:
        Conv, Pool, GAP = layers.Conv2D, layers.MaxPooling2D, layers.GlobalAveragePooling2D
        k_init, k_res = (5, 5), (3, 3)
    elif spatial_dims == 3:
        Conv, Pool, GAP = layers.Conv3D, layers.MaxPooling3D, layers.GlobalAveragePooling3D
        k_init, k_res = (3, 3, 3), (3, 3, 3)
    else:
        Conv, Pool, GAP = layers.Conv1D, layers.MaxPooling1D, layers.GlobalAveragePooling1D
        k_init, k_res = 5, 3

    # --- Model Definition ---
    inputs = keras.Input(shape=input_shape)
    
    # Initial Block
    x = Conv(32, k_init, padding='same', kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    try:
        x = Pool(2)(x)
    except:
        pass # Handle cases where dimension is too small to pool
    
    # Each residual_block adds 2 conv layers.
    num_blocks = max(1, (num_conv_layers - 2) // 2)
    
    filters = 32
    for i in range(num_blocks):
        # Expand filters and reduce resolution every 3 blocks
        if i > 0 and i % 3 == 0:
            filters = min(filters * 2, 256)
            stride = 2
        else:
            stride = 1
            
        x = residual_block(x, filters, reg, Conv, k_res, stride=stride)
    
    x = GAP()(x)
    
    # Dense Head
    dense_units = params.get("dense_units", 128)
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def residual_block(x, filters, reg, ConvLayer, k_size, stride=1):
    shortcut = x
    
    x = ConvLayer(filters, k_size, strides=stride, padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = ConvLayer(filters, k_size, strides=1, padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = ConvLayer(filters, 1, strides=stride, padding='same', kernel_regularizer=reg)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x