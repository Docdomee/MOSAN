def build_model(input_shape, num_classes, params):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    
    # Hyperparameters from "Stabilized Deep 3D Conv" design strategy
    dropout_rate = params.get("dropout_rate", 0.7)      # Target 0.68-0.72 range
    learning_rate = params.get("learning_rate", 0.0007)  # Conservative optimization
    weight_decay = params.get("weight_decay", 0.005)     # Regularization stack
    dense_units = params.get("dense_units", 256)         # Optimal capacity
    num_conv_layers = 6                                  # Fixed depth per strategy
    num_filters = 64                                     # Fixed width per strategy
    se_reduction = params.get("se_reduction", 16)        # SE block reduction ratio
    
    # Validate input dimensions can support 6 pooling layers (factor of 64 reduction)
    min_dim = min(input_shape[1], input_shape[2])
    max_layers = int(np.log2(min_dim)) if min_dim > 0 else 1
    if num_conv_layers > max_layers:
        print(f"[Builder Warning] Requested {num_conv_layers} layers, but spatial dims only support {max_layers}. "
              f"Consider increasing input spatial resolution or reducing num_conv_layers.")
        num_conv_layers = max_layers
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Squeeze-and-Excitation block for channel recalibration in 3D
    def se_block(input_tensor, channels, reduction=16):
        # Squeeze: Global average pooling across spatial and temporal dims
        se = keras.layers.GlobalAveragePooling3D()(input_tensor)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        se = keras.layers.Reshape((1, 1, 1, channels))(se)
        se = keras.layers.Dense(channels // reduction, activation='relu', use_bias=False)(se)
        se = keras.layers.Dense(channels, activation='sigmoid', use_bias=False)(se)
        # Scale
        return keras.layers.Multiply()([input_tensor, se])
    
    # 6-layer 3D CNN with residual connections and SE blocks
    for i in range(num_conv_layers):
        shortcut = x
        
        # Main convolutional path
        x = keras.layers.Conv3D(
            num_filters,
            kernel_size=3,
            padding="same",
            use_bias=False  # BN handles bias
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Channel attention via Squeeze-and-Excitation
        x = se_block(x, num_filters, se_reduction)
        
        # Residual connection with projection if needed
        if shortcut.shape[-1] != num_filters:
            shortcut = keras.layers.Conv3D(
                num_filters,
                kernel_size=1,
                padding="same",
                use_bias=False
            )(shortcut)
        
        x = keras.layers.Add()([x, shortcut])
        
        # Spatial pooling: preserve temporal (dim 0), reduce spatial (dims 1,2)
        x = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # Global pooling and classification head
    x = keras.layers.GlobalAveragePooling3D()(x)
    
    # Dense block with regularization stack (dropout + weight decay via optimizer)
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    
    x = keras.layers.Dense(num_classes)(x)
    outputs = keras.layers.Activation("softmax", dtype="float32")(x)
    
    # AdamW optimizer for decoupled weight decay (avoiding L2 regularization on layers)
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model