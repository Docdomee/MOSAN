def build_model(input_shape, num_classes, params: dict):
    """
    Optimized 3D CNN for 3D_DYNAMIC_GAF representations.
    
    Addresses overfitting crisis via Shallow-Wide architecture (3 layers, 64 filters)
    and aggressive double regularization (Dropout 0.55 + Weight Decay 0.008).
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    
    def _get_regularizer(wd):
        if wd > 0:
            return keras.regularizers.l2(wd)
        return None
    
    # Architecture: Strictly 3-4 layers (Shallow-Wide motif)
    min_dim = min(input_shape[1], input_shape[2])
    max_layers = int(np.log2(min_dim)) if min_dim > 0 else 1
    requested_layers = params.get("num_conv_layers", 3)  # Default 3 per design strategy
    num_conv_layers = min(requested_layers, max_layers, 4)  # Hard cap at 4
    
    # Width: 64 filters constant (avoid exponential growth to prevent overfitting)
    base_filters = params.get("filters", 64)
    
    # Regularization Stack (Critical for 3D_DYNAMIC_GAF)
    dropout_rate = params.get("dropout_rate", 0.55)  # Optimal range 0.5-0.6
    weight_decay = params.get("weight_decay", 0.008)  # Aggressive: 0.005-0.01 range
    reg = _get_regularizer(weight_decay)
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Convolutional Stack: 3-4 layers, constant 64 filters, kernel size 3
    for i in range(num_conv_layers):
        # Maintain constant width (64) rather than doubling (2**i) to prevent capacity overfitting
        num_filters = base_filters
        
        shortcut = x
        
        # Conv3D with L2 regularization and no bias (BatchNorm handles it)
        x = keras.layers.Conv3D(
            num_filters,
            kernel_size=params.get("kernel_size", 3),
            padding="same",
            kernel_regularizer=reg,
            use_bias=False
        )(x)
        
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Optional residual (disabled by default to reduce complexity)
        if params.get("use_residual", False):
            if shortcut.shape[-1] != num_filters:
                shortcut = keras.layers.Conv3D(
                    num_filters,
                    kernel_size=1,
                    padding="same",
                    kernel_regularizer=reg
                )(shortcut)
            x = keras.layers.Add()([x, shortcut])
        
        # Spatial pooling only (preserve temporal dimension)
        x = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # Global pooling and Dense classifier
    x = keras.layers.GlobalAveragePooling3D()(x)
    
    # Dense: 128 default (reduce to 64 via params if overfitting persists)
    dense_units = params.get("dense_units", 128)
    x = keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=reg
    )(x)
    
    # Dropout 0.5-0.6 range (0.55 default per Critical Analysis)
    x = keras.layers.Dropout(dropout_rate)(x)
    
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    
    # Learning rate: 0.001 standard or 0.0005 for finer convergence
    lr = params.get("learning_rate", 0.001)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model