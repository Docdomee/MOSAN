import tensorflow as tf
from tensorflow.keras import layers, Model

def create_1d_cnn_model(input_shape, num_classes, num_conv_layers=6, filters=64, kernel_size=7, dense_units=128, dropout_rate=0.4):
    """Create a 1D CNN model for SERS spectra classification"""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv1D(filters, kernel_size, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    # Additional convolutional layers
    for i in range(num_conv_layers - 1):
        x = layers.Conv1D(filters * (2 if i < 2 else 4), kernel_size, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
    
    # Global pooling and dense layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units // 2, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# Example usage
if __name__ == "__main__":
    model = create_1d_cnn_model(input_shape=(2048, 1), num_classes=5)
    model.summary()