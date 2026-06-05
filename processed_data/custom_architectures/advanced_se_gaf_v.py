def build_model(input_shape, num_classes, params):
    """
    Advanced SE-ResNet for GAF (Gramian Angular Field) Classification.
    
    Architecture optimized based on Pattern Analysis:
    - 7-layer SE-ResNet depth (7 residual blocks)
    - ~1.2M parameters
    - Aggressive Learning Rate support (1.5e-4 to 6e-4)
    - Calibrated Regularization: Dropout [0.42, 0.52] + L2 [1.2e-4, 2.5e-4]
    - Squeeze-and-Excitation blocks for channel attention
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # --- Hyperparameters from Pattern Scout Analysis ---
    # Priority 1: Learning Rate Optimization supported via architecture stability (BN, Residuals)
    # Priority 2: Regularization Calibration
    dropout_rate = params.get("dropout_rate", 0.48)  # Optimal zone: [0.42, 0.52]
    l2_lambda = params.get("l2_reg", 1.8e-4)        # Optimal zone: [1.2e-4, 2.5e-4]
    se_reduction = params.get("se_reduction", 16)    # SE block reduction ratio
    
    # Exploration Vector support: High-LR + High-Reg frontier
    # If dropout > 0.55, caller should increase L2 to >3e-4 via params
    regularizer = keras.regularizers.l2(l2_lambda)
    
    def squeeze_excite_block(input_tensor, filters, ratio=16):
        """
        Squeeze-and-Excitation block.
        Learns channel-wise attention to improve representational power.
        """
        se = layers.GlobalAveragePooling2D(keepdims=True)(input_tensor)
        se = layers.Dense(filters // ratio, activation='relu', 
                         kernel_regularizer=regularizer, use_bias=False)(se)
        se = layers.Dense(filters, activation='sigmoid', 
                         kernel_regularizer=regularizer, use_bias=False)(se)
        return layers.Multiply()([input_tensor, se])
    
    def residual_se_block(x, filters, kernel_size=3, stride=1, use_projection=False):
        """
        Residual block with integrated Squeeze-and-Excitation.
        Supports aggressive learning rates through residual connections and BN.
        """
        shortcut = x
        
        # First conv layer
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                         kernel_regularizer=regularizer, kernel_initializer='he_normal', 
                         use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second conv layer
        x = layers.Conv2D(filters, kernel_size, padding='same',
                         kernel_regularizer=regularizer, kernel_initializer='he_normal', 
                         use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        
        # SE attention mechanism (channel recalibration)
        x = squeeze_excite_block(x, filters, ratio=se_reduction)
        
        # Projection shortcut if dimensions change
        if use_projection or stride != 1:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                    kernel_regularizer=regularizer, 
                                    kernel_initializer='he_normal', use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Residual connection
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x
    
    # --- Input Layer ---
    inputs = keras.Input(shape=input_shape)
    
    # --- Initial Convolution ---
    x = layers.Conv2D(48, 3, padding='same', kernel_regularizer=regularizer,
                     kernel_initializer='he_normal', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # --- 7-Layer SE-ResNet Blocks ---
    # Stage 1: 2 blocks, 48 filters (maintains spatial dimensions)
    x = residual_se_block(x, 48, use_projection=True)
    x = residual_se_block(x, 48)
    
    # Stage 2: 2 blocks, 96 filters (downsampling via stride=2 on first block)
    x = residual_se_block(x, 96, stride=2, use_projection=True)
    x = residual_se_block(x, 96)
    
    # Stage 3: 3 blocks, 128 filters (downsampling via stride=2 on first block)
    # This achieves the 7-layer depth while maintaining ~1.2M parameters
    x = residual_se_block(x, 128, stride=2, use_projection=True)
    x = residual_se_block(x, 128)
    x = residual_se_block(x, 128)
    
    # --- Classification Head ---
    x = layers.GlobalAveragePooling2D()(x)
    
    # Primary dropout (Pattern: 0.42-0.52 optimal for generalization)
    x = layers.Dropout(dropout_rate)(x)
    
    # Dense hidden layer with regularization
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizer)(x)
    
    # Secondary dropout (scaled for penultimate layer per best practices)
    x = layers.Dropout(min(dropout_rate * 0.85, 0.6))(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', 
                          kernel_regularizer=regularizer)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Advanced_SE_GAF_v2")
    return model