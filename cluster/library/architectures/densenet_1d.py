import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, params):
    """
    DenseNet-1D Architecture (Hackable Baseline).
    
    Features:
    - Dense connectivity: Each layer receives inputs from all preceding layers.
    - Growth Rate (k): Controls how much information is added at each layer.
    - Transition Blocks: Compress features (1x1 conv) and downsample (pool).
    """
    
    # --- Params ---
    growth_rate = 32
    num_blocks_per_stage = 4
    compression_factor = 0.5
    dropout_rate = params.get("dropout_rate", 0.2)
    
    l2_reg = 1e-4
    reg = keras.regularizers.l2(l2_reg)

    # --- Helpers ---
    def conv_block(x, growth_rate):
        """BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3) [Bottleneck Design]"""
        x1 = layers.BatchNormalization()(x)
        x1 = layers.Activation("relu")(x1)
        x1 = layers.Conv1D(4 * growth_rate, 1, use_bias=False, kernel_regularizer=reg)(x1) # Bottleneck
        
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation("relu")(x1)
        x1 = layers.Conv1D(growth_rate, 3, padding="same", use_bias=False, kernel_regularizer=reg)(x1)
        
        if dropout_rate > 0:
            x1 = layers.Dropout(dropout_rate)(x1)
            
        return layers.Concatenate()([x, x1])

    def transition_block(x, reduction):
        """1x1 Conv + AvgPool to reduce size and channels"""
        num_filters = int(x.shape[-1] * reduction)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv1D(num_filters, 1, use_bias=False, kernel_regularizer=reg)(x)
        x = layers.AveragePooling1D(2, strides=2)(x)
        return x

    # --- Main Build ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Initial Conv
    x = layers.Conv1D(64, 7, strides=2, padding="same", use_bias=False, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(3, strides=2, padding="same")(x)
    
    # Dense Blocks
    # 3 Stages
    for i in range(3):
        # Dense Block
        for _ in range(num_blocks_per_stage):
            x = conv_block(x, growth_rate)
            
        # Transition Layer (except after last block)
        if i < 2:
            x = transition_block(x, compression_factor)
            
    # Head
    x = layers.GlobalAveragePooling1D()(x)
    layers.Dense(num_classes)(x)`n    outputs = layers.Activation("softmax", dtype="float32")(x)
    
    model = keras.Model(inputs, outputs, name="DenseNet_1D")
    
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
