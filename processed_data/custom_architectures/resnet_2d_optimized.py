import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape, num_classes, hyperparameters=None, **kwargs):
    """ResNet-inspired 2D CNN architecture for SERS spectral classification"""
    
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(32, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    def residual_block(x, filters, strides=1):
        shortcut = x
        
        # First convolution
        x = layers.Conv2D(filters, 3, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second convolution
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Shortcut connection
        if strides > 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=strides)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x
    
    # Stack residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, strides=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, strides=2)
    x = residual_block(x, 256)
    
    # Global average pooling and output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Example usage
if __name__ == "__main__":
    model = build_model(input_shape=(48, 48, 1), num_classes=10)
    model.summary()