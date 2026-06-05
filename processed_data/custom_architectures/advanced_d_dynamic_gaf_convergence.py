def build_model(input_shape, num_classes, params: dict):
    """
    Advanced 3D Dynamic GAF CNN implementing the 5-Layer Convergence Zone strategy.
    
    Key improvements:
    - 5-layer optimal depth configuration (avoiding the 4-layer dead zone)
    - 32-filter efficiency anchor with gentle growth option
    - Adaptive regularization: weight decay (0.001-0.01) + dropout 0.6
    - Cosine annealing LR schedule for deep networks (depth >=5)
    - Larger kernel sizes (5) in early layers for temporal dependency capture
    - Residual connections enabled by default for gradient flow in deep towers
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    
    # Determine feasible layer depth based on spatial dimensions
    min_dim = min(input_shape[1], input_shape[2])
    max_layers = int(np.log2(min_dim)) if min_dim > 0 else 1
    
    # Default to 5-layer convergence zone per strategy; avoid 4-layer dead zone
    requested_layers = params.get("num_conv_layers", 5)
    num_conv_layers = min(requested_layers, max_layers)
    
    if num_conv_layers < requested_layers:
        print(
            f"[Builder Warning] 3D_CNN: Requested {requested_layers} layers, "
            f"but data dimension only supports {num_conv_layers}. Adjusting automatically."
        )
    
    # Architectural hyperparameters from 5-Layer Convergence strategy
    base_filters = params.get("filters", 32)  # 32 filters: optimal accuracy-to-parameter ratio
    dropout_rate = params.get("dropout_rate", 0.6)  # 0.6 optimal for 5-layer (anti-pattern: >0.8)
    learning_rate = params.get("learning_rate", 0.001)  # Base LR for mid-depth networks
    weight_decay = params.get("weight_decay", 0.001)  # Range: 0.001-0.01 for regularization
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Convolutional tower: 5-Layer Convergence Zone implementation
    for i in range(num_conv_layers):
        # Filter strategy: Constant 32 recommended, or gentle growth to prevent over-parameterization
        filter_strategy = params.get("filter_strategy", "constant")
        if filter_strategy == "constant":
            num_filters = base_filters
        elif filter_strategy == "gentle":
            # 32, 32, 64, 64, 128 pattern for 5 layers (avoids 4.3M params of aggressive doubling)
            num_filters = min(512, base_filters * (2 ** (i // 2)))
        else:
            # Legacy doubling (use with caution - high parameter count)
            num_filters = min(512, base_filters * (2 ** i))
        
        # Kernel size strategy: Larger receptive fields in early layers for 17-length history
        # Captures longer temporal dependencies in GAF representation
        if i < params.get("large_kernel_layers", 2):
            kernel_size = params.get("early_kernel_size", 5)  # kernel_size=5 for temporal context
        else:
            kernel_size = params.get("kernel_size", 3)
        
        # Residual shortcut for gradient flow (critical for 5+ layers)
        shortcut = x
        
        # Conv3D block with L2 weight decay (adaptive regularization)
        x = keras.layers.Conv3D(
            num_filters,
            kernel_size=kernel_size,
            padding="same",
            kernel_regularizer=keras.regularizers.l2(weight_decay)
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Residual connection (enabled by default for deep networks)
        if params.get("use_residual", True):
            if shortcut.shape[-1] != num_filters:
                shortcut = keras.layers.Conv3D(
                    num_filters,
                    kernel_size=1,
                    padding="same",
                    kernel_regularizer=keras.regularizers.l2(weight_decay)
                )(shortcut)
            x = keras.layers.Add()([x, shortcut])
        
        # Spatial pooling only (1, 2, 2) to preserve temporal dimension
        x = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # Global feature extraction
    x = keras.layers.GlobalAveragePooling3D()(x)
    
    # Dense classifier with regularization
    x = keras.layers.Dense(
        params.get("dense_units", 128),
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(weight_decay)
    )(x)
    
    # Critical: 0.6 dropout for 5-layer (strategy warns against >0.8 catastrophic underfitting)
    x = keras.layers.Dropout(dropout_rate)(x)
    
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    
    # Optimizer with cosine annealing for depth >= 5 (per depth-regularization scaling law)
    if num_conv_layers >= 5 and params.get("use_cosine_decay", True):
        decay_epochs = params.get("decay_epochs", 100)
        alpha = params.get("alpha", 0.01)  # Minimum LR fraction
        
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_epochs,
            alpha=alpha
        )
        
        # Prefer AdamW for explicit weight decay separation, fallback to Adam
        if hasattr(keras.optimizers, 'AdamW'):
            optimizer = keras.optimizers.AdamW(
                learning_rate=lr_schedule,
                weight_decay=weight_decay
            )
        else:
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        # Shallow networks (3 layers): Standard Adam with moderate LR
        if hasattr(keras.optimizers, 'AdamW'):
            optimizer = keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
        else:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model