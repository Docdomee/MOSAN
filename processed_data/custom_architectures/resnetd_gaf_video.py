def build_model(input_shape, num_classes, params):
    """
    3D ResNet architecture tailored for GAF-based video classification.
    
    This model implements a 3D ResNet architecture with:
    - Progressive filter growth (32 -> 64 -> 128 -> 256)
    - Residual connections with 1x1x1 convolutions for dimension matching
    - Batch normalization and ReLU activation
    - Global average pooling for spatial-temporal feature aggregation
    - Dense layers with dropout for final classification
    
    Args:
        input_shape: Tuple of (frames, height, width, channels)
        num_classes: Number of output classes
        params: Dictionary of model hyperparameters
        
    Returns:
        Compiled Keras model
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # Initial convolution layer
    x = layers.Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # ResNet blocks
    # Block 1
    x = _resnet_block(x, 32, 3)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    # Block 2
    x = _resnet_block(x, 64, 3)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    # Block 3
    x = _resnet_block(x, 128, 3)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    # Block 4
    x = _resnet_block(x, 256, 3)
    
    # Global average pooling
    x = layers.GlobalAveragePooling3D()(x)
    
    # Dense layers
    x = layers.Dense(params.get("dense_units", 512), activation='relu')(x)
    x = layers.Dropout(params.get("dropout_rate", 0.5))(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    # Create model
    model = keras.Model(inputs, outputs)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def _resnet_block(x, filters, kernel_size):
    """Create a ResNet block with 3D convolutions."""
    from tensorflow.keras import layers
    
    # First conv layer
    shortcut = x
    x = layers.Conv3D(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second conv layer
    x = layers.Conv3D(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Adjust shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv3D(filters, kernel_size=1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add shortcut
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x