import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def build_model(input_shape, num_classes, params):
    """
    Fixed Weight 1D CNN with Extreme Learning Machine approach
    Convolutional weights are fixed after initialization
    Only the final classification layer is trainable
    
    Args:
        input_shape: Tuple of (timesteps, features)
        num_classes: Number of output classes
        params: Dictionary of hyperparameters for compatibility
    """
    
    # Extract parameters or use defaults
    num_conv_layers = params.get('num_conv_layers', 3)
    filters = params.get('filters', 32)
    kernel_size = params.get('kernel_size', 5)
    dense_units = params.get('dense_units', 64)
    dropout_rate = params.get('dropout_rate', 0.3)
    
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    
    # Fixed convolutional layers
    x = inputs
    for i in range(num_conv_layers):
        # Fixed convolutional layer (no bias to keep weights fixed)
        x = layers.Conv1D(
            filters=filters * (2 ** i),  # Increasing filters
            kernel_size=kernel_size,
            padding='same',
            use_bias=False,  # No bias to maintain fixed weights
            trainable=False,  # Fixed weights
            kernel_initializer='glorot_uniform'
        )(x)
        
        # Fixed batch normalization
        x = layers.BatchNormalization(trainable=False)(x)
        
        # Fixed activation
        x = layers.Activation('relu')(x)
        
        # Fixed pooling
        if i < num_conv_layers - 1:  # No pooling after last conv layer
            x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Feature mapping layers (fixed weights)
    x = layers.Dense(dense_units, activation='relu', trainable=False)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Trainable output layer (Extreme Learning Machine approach)
    outputs = layers.Dense(num_classes, activation='softmax', trainable=True)(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile with fixed learning rate for the output layer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# For compatibility with validation system
if __name__ == "__main__":
    # Test the build function
    dummy_shape = (128, 1)
    dummy_params = {
        'num_conv_layers': 3,
        'filters': 32,
        'kernel_size': 5,
        'dense_units': 64,
        'dropout_rate': 0.3
    }
    
    model = build_model(dummy_shape, 10, dummy_params)
    model.summary()