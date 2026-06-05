def build_model(input_shape, num_classes, params: dict):
    """
    Advanced 2D GAF CNN with Deep Residual Architecture.
    
    Implements Design Strategy optimizations:
    - Deep architecture (6-8 conv layers) for hierarchical GAF structure capture
    - Kernel 7 for early layers (angular correlation capture), kernel 3 for refinement
    - Aggressive regularization (dropout 0.50-0.55, L2 reg)
    - Residual connections for gradient flow in deep networks
    - Optimized for GAF temporal-spatial pattern recognition
    
    Representation: '2D_GAF'
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Defensive cast
    input_shape = tuple(input_shape)
    
    # --- Input Validation for 2D_GAF Representation ---
    # Ensure input has channel dimension (H, W, C) not just (H, W)
    if len(input_shape) == 2:
        # 2D_GAF single channel: reshape (H, W) -> (H, W, 1)
        needs_reshape = True
        working_shape = (*input_shape, 1)
    else:
        needs_reshape = False
        working_shape = input_shape
    
    # --- Architecture Depth Configuration (Design Strategy Phase 1) ---
    min_dim = min(working_shape[0], working_shape[1])
    max_layers_calc = int(np.log2(min_dim)) if min_dim > 0 else 1
    
    # Design Strategy: Minimum 6 layers required, 7-8 optimal for GAF
    requested_layers = params.get("num_conv_layers", 7)
    num_conv_layers = min(requested_layers, max_layers_calc)
    
    # Ensure at least 1 layer to prevent empty network
    num_conv_layers = max(1, num_conv_layers)
    
    if num_conv_layers < 6:
        print(f"[Advanced GAF Warning] Only {num_conv_layers} layers possible. Design Strategy recommends 6+ layers for GAF.")
    
    # --- Design Strategy Parameters (Phase 3 Regularization Stack) ---
    base_filters = params.get("filters", 32)  # Deep-Narrow motif: 32 filters efficient
    dense_units = params.get("dense_units", 256)  # DS: >= 256 required
    dropout_rate = params.get("dropout_rate", 0.50)  # DS: 0.50-0.55 optimal
    learning_rate = params.get("learning_rate", 1e-4)  # DS: <= 0.0005, target 1e-4
    use_residual = params.get("use_residual", True)  # DS: Essential for 6+ layers
    reg_lambda = params.get("regularization", 1e-4)
    
    # Setup regularizer
    reg = keras.regularizers.l2(reg_lambda) if reg_lambda > 0 else None
    
    # --- Model Construction ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Handle 2D_GAF channel expansion if needed
    if needs_reshape:
        x = layers.Reshape(working_shape)(x)
    
    # Data Augmentation (2D) - maintained from base
    if params.get("use_data_augmentation"):
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
    
    # --- Convolutional Blocks with Design Strategy Optimizations ---
    for i in range(num_conv_layers):
        # Phase 2: Receptive Field Optimization
        # First 3 layers: kernel 7 for GAF angular correlation capture
        # Deeper layers: kernel 3 for fine-grained refinement
        if i < 3:
            kernel_size = 7
        else:
            kernel_size = 3
            
        # Filter progression: Deep-Narrow motif
        # Grow exponentially but cap at 256 to avoid overfitting (per DS)
        num_filters = min(256, base_filters * (2 ** i))
        
        # Residual shortcut - capture before any transformations
        shortcut = x
        
        # Main convolution path
        x = layers.Conv2D(
            num_filters,
            (kernel_size, kernel_size),
            padding="same",
            kernel_regularizer=reg,
            use_bias=False  # BatchNorm handles bias
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Design Strategy: For deep networks (6+ layers), add secondary conv for feature refinement
        if num_conv_layers >= 6 and i >= 3:
            x = layers.Conv2D(
                num_filters,
                (3, 3),
                padding="same",
                kernel_regularizer=reg,
                use_bias=False
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        
        # Residual connection (essential for 6+ layer GAF networks)
        if use_residual:
            # Check if channel dimensions match for addition
            shortcut_filters = int(shortcut.shape[-1]) if shortcut.shape[-1] is not None else num_filters
            if shortcut_filters != num_filters:
                shortcut = layers.Conv2D(
                    num_filters, 
                    (1, 1), 
                    padding="same",
                    kernel_regularizer=reg,
                    use_bias=False,
                    name=f"projection_shortcut_{i}"
                )(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            x = layers.Add(name=f"residual_add_{i}")([x, shortcut])
        
        # Pooling: Progressive feature extraction
        # Don't pool on last conv layer to preserve spatial information for GAP
        if i < num_conv_layers - 1:
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # --- Classification Head (Phase 3 Regularization) ---
    x = layers.GlobalAveragePooling2D()(x)
    
    # Design Strategy: Dense units >= 256
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=reg
    )(x)
    
    # Design Strategy: Aggressive dropout 0.50-0.55
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    # --- Model Compilation ---
    model = keras.Model(inputs, outputs)
    
    # Design Strategy: Learning rate <= 0.0005
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model