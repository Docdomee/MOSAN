# filename: deep_1d_resnet_cnn.py
# representation: 1D_CNN
# manifest: NPY_nature_30_classes
# library_choice: resnet_1d.py

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Model


def build_model(input_shape, num_classes, params=None):
    """
    Build a 1D CNN ResNet model for the specified input shape and number of classes.
    
    Args:
        input_shape: Shape of the input data (sequence_length, features)
        num_classes: Number of output classes
        params: Optional dictionary of model hyperparameters
        
    Returns:
        Compiled Keras model
    """
    
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolution layer
    x = Conv1D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Residual blocks would go here
    # For brevity, showing a simple residual block structure
    
    # Example residual block
    shortcut = x
    x = Conv1D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    
    # Global average pooling and output
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model