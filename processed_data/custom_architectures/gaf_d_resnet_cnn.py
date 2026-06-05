import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_model(input_shape, num_classes, params):
    """
    ResNet-style 2D CNN Architecture for GAF (Gramian Angular Field) Images.
    
    This model is designed specifically for 2D GAF representations of SERS spectra.
    It incorporates residual connections, batch normalization, and a deep convolutional
    structure based on insights from hyperparameter optimization trials.
    
    Key Design Choices:
    - Depth: 8 convolutional layers organized in residual blocks
    - Width: Filter counts from 64 to 512
    - Regularization: Moderate dropout (0.25) and weight decay (1e-4)
    - Optimization: Adam with learning rate 1e-4
    """
    
    # --- Hyperparameters ---
    reg = keras.regularizers.l2(params.get("weight_decay", 1e-4))
    dropout_rate = params.get("dropout_rate", 0.25)
    learning_rate = params.get("learning_rate", 1e-4)
    use_residual = params.get("use_residual", True)
    
    # --- Input ---
    inputs = keras.Input(shape=input_shape)
    
    # --- Initial Convolution ---
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    # --- Residual Block Definition ---
    def residual_block(x, filters, stride=1, conv_shortcut=False):
        """Creates a residual block with two 3x3 convolutions."""
        shortcut = x
        
        # First convolution
        x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same', kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second convolution
        x = layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        
        # Shortcut connection
        if conv_shortcut:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, kernel_regularizer=reg)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Add shortcut to output
        if use_residual:
            x = layers.Add()([x, shortcut])
        else:
            x = shortcut  # In case we want to disable residuals for ablation
        
        x = layers.Activation('relu')(x)
        return x
    
    # --- ResNet-like Architecture ---
    # Stage 1
    x = residual_block(x, 64, conv_shortcut=True)
    x = residual_block(x, 64)
    
    # Stage 2
    x = residual_block(x, 128, stride=2, conv_shortcut=True)
    x = residual_block(x, 128)
    
    # Stage 3
    x = residual_block(x, 256, stride=2, conv_shortcut=True)
    x = residual_block(x, 256)
    
    # Stage 4
    x = residual_block(x, 512, stride=2, conv_shortcut=True)
    x = residual_block(x, 512)
    
    # --- Global Average Pooling ---
    x = layers.GlobalAveragePooling2D()(x)
    
    # --- Classification Head ---
    x = layers.Dense(params.get("dense_units", 256), activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    # --- Model Compilation ---
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model