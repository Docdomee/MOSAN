import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, params):
    """
    ResNet-18 2D Architecture (Hackable Baseline).
    
    Standard ResNet adaptation for 2D inputs (Spectrograms, Images).
    Uses the classic BasicBlock (2 convs).
    """

    reg = keras.regularizers.l2(1e-4)

    def residual_block_2d(x, filters, stride=1):
        shortcut = x
        
        # Conv 1
        x = layers.Conv2D(filters, 3, strides=stride, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Conv 2
        x = layers.Conv2D(filters, 3, strides=1, padding="same", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        
        # Shortcut connection
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", kernel_regularizer=reg)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
            
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        return x

    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Stem
    x = layers.Conv2D(64, 7, strides=2, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    # ResNet-18 Structure: [2, 2, 2, 2] blocks
    # Stage 1
    x = residual_block_2d(x, 64)
    x = residual_block_2d(x, 64)
    
    # Stage 2
    x = residual_block_2d(x, 128, stride=2)
    x = residual_block_2d(x, 128)
    
    # Stage 3
    x = residual_block_2d(x, 256, stride=2)
    x = residual_block_2d(x, 256)
    
    # Stage 4
    x = residual_block_2d(x, 512, stride=2)
    x = residual_block_2d(x, 512)
    
    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(params.get("dropout_rate", 0.5))(x)
    layers.Dense(num_classes)(x)`n    outputs = layers.Activation("softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="ResNet18_2D")
    
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
