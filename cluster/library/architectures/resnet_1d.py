import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, params):
    """
    ResNet-1D Architecture (Hackable Baseline).
    
    A clean, modular implementation of a Residual Network adapted for 1D time-series data.
    Structure:
    - Initial Conv Block
    - Stack of Residual Blocks
    - Global Average Pooling
    - Classification Head
    
    Key Features for Innovation:
    - Explicit `residual_block` function to allow easy modification of the internal structure.
    - Uses 'he_normal' initialization standard for ReLUs.
    """
    
    # --- Configuration ---
    # These can be tuned by the Innovation Team or Params
    filters_list = [64, 128, 256, 512] # Depth of the network
    kernel_size = params.get("kernel_size", 3)
    dropout_rate = params.get("dropout_rate", 0.3)
    
    # Regularization
    l2_reg = 1e-4 if params.get("use_l2_regularization") else 0.0
    reg = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    # --- Helper: Residual Block ---
    def residual_block(x, filters, kernel_size, stride=1):
        shortcut = x
        
        # First Conv
        x = layers.Conv1D(filters, kernel_size, strides=stride, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Second Conv
        x = layers.Conv1D(filters, kernel_size, strides=1, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        
        # Projection Shortcut (if dimensions change)
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, strides=stride, padding="same", kernel_regularizer=reg)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
            
        # Add & Activate
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        return x

    # --- Model Definition ---
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # 1. Initial Convolution (Stem)
    x = layers.Conv1D(64, 7, strides=2, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(3, strides=2, padding="same")(x)

    # 2. Residual Stages
    # We stack blocks. For a deeper net, loop more times per filter size.
    for filters in filters_list:
        # First block does downsampling (stride=2), others keep stride=1
        x = residual_block(x, filters, kernel_size, stride=2)
        x = residual_block(x, filters, kernel_size, stride=1) 

    # 3. Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    layers.Dense(num_classes)(x)`n    outputs = layers.Activation("softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="ResNet_1D")
    
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    return model
