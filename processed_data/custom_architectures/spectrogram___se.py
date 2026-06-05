def build_model(input_shape, num_classes, params: dict):
    """
    Goldilocks 2D Spectrogram CNN with Residual Connections and Squeeze-Excitation.
    
    Architecture based on the "6-128-3" motif:
    - 6 Convolutional layers arranged in 3 Residual Blocks (2 layers per block)
    - Kernel size 3x3 (locked, per anti-pattern analysis)
    - Filter modes: 'flat' [128,128,128] or 'progressive' [64,128,256]
    - Squeeze-and-Excitation blocks after each residual block (reduction ratio 16)
    - Conservative regularization: Dropout 0.5, Dense units 256 (avoid 1024)
    - Low learning rate 1e-5 for stable convergence
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np
    
    def _get_regularizer(p):
        l2_val = p.get('l2_reg', 1e-4)
        return keras.regularizers.l2(l2_val) if l2_val > 0 else None
    
    def se_block(inputs, reduction_ratio=16):
        """Squeeze-and-Excitation block for channel-wise recalibration."""
        filters = inputs.shape[-1]
        se = layers.GlobalAveragePooling2D()(inputs)
        se = layers.Reshape((1, 1, filters))(se)
        se = layers.Dense(filters // reduction_ratio, activation='relu', use_bias=False, 
                         kernel_initializer='he_normal')(se)
        se = layers.Dense(filters, activation='sigmoid', use_bias=False,
                         kernel_initializer='he_normal')(se)
        return layers.multiply([inputs, se])
    
    def residual_block_2conv(x, filters, reg, use_se=True, se_ratio=16):
        """
        Residual block containing 2 Conv layers (kernel 3x3).
        Implements skip connection every 2 layers as per strategy.
        """
        shortcut = x
        
        # First conv
        x = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=reg,
                         kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second conv
        x = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=reg,
                         kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        
        # Projection shortcut if channel mismatch
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding='same', kernel_regularizer=reg,
                                   kernel_initializer='he_normal')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        
        # Residual addition
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        # Squeeze-and-Excitation
        if use_se:
            x = se_block(x, reduction_ratio=se_ratio)
        
        return x
    
    # Validate input dimensions for 6 layers (3 blocks with MaxPool)
    min_dim = min(input_shape[0], input_shape[1])
    max_blocks = int(np.log2(min_dim)) - 1 if min_dim >= 8 else 0
    target_blocks = 3  # 6 conv layers total
    
    if max_blocks < target_blocks:
        print(f"[Architect Warning] Input {input_shape} supports only {max_blocks} blocks. "
              f"Goldilocks architecture requires {target_blocks} blocks (6 layers).")
        num_blocks = max(1, max_blocks)
    else:
        num_blocks = target_blocks
    
    # Filter schedule: 'flat' (Goldilocks 128) or 'progressive' [64,128,256]
    filter_mode = params.get('filter_growth_mode', 'flat')
    if filter_mode == 'progressive':
        # Note: 256 exceeds the 192 avoidance limit but is within parameter budget (~4M)
        block_filters = [64, 128, 256][:num_blocks]
    else:
        # Default Goldilocks: flat 128 filters
        block_filters = [128, 128, 128][:num_blocks]
    
    # Build model
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Data Augmentation (2D)
    if params.get('use_data_augmentation'):
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
    
    reg = _get_regularizer(params)
    use_se = params.get('use_se', True)
    se_ratio = params.get('se_reduction', 16)
    
    # 3 Residual blocks = 6 Conv layers (kernel 3 locked)
    for i, filters in enumerate(block_filters):
        x = residual_block_2conv(x, filters, reg, use_se=use_se, se_ratio=se_ratio)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layer: 256-512 range (avoid 1024 per anti-pattern)
    dense_units = params.get('dense_units', 256)
    if dense_units > 512:
        print(f"[Architect Warning] dense_units={dense_units} exceeds safe limit of 512.")
    
    x = layers.Dense(dense_units, activation='relu', kernel_regularizer=reg,
                    kernel_initializer='he_normal')(x)
    
    # Conservative dropout: exactly 0.5 (avoid >0.55 per anti-pattern)
    dropout_rate = params.get('dropout_rate', 0.5)
    if dropout_rate > 0.55:
        print(f"[Architect Warning] dropout_rate={dropout_rate} may destabilize training.")
    
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Low learning rate 1e-5 for stable convergence (avoid >2e-5)
    lr = params.get('learning_rate', 1e-5)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model