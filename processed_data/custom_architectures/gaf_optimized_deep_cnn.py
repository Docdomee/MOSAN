def build_model(input_shape, num_classes, params):
    """
    Optimized Deep CNN for 2D GAF (Gramian Angular Field) representations.
    
    Incorporates Pattern Scout findings:
    - Deep architecture (6-8 layers) with residual connections
    - Narrow filter strategy (48-64 constant) or controlled growth
    - Large kernel size (7x7) for capturing angular correlations
    - Optimized dense layer (256 units) with moderate dropout (0.45-0.50)
    - Optional Squeeze-and-Excitation (SE) blocks for channel attention
    
    Args:
        input_shape: Tuple (height, width, channels)
        num_classes: Number of output classes
        params: Dict containing:
            - num_conv_layers: int (default 6, Pattern Scout recommends 6-7)
            - filters: int (default 48, Pattern Scout recommends 48-64)
            - kernel_size: int (default 7, Pattern Scout recommends 7)
            - dense_units: int (default 256, Pattern Scout sweet spot)
            - dropout_rate: float (default 0.5, Pattern Scout recommends 0.45-0.50)
            - use_constant_filters: bool (default True, keeps narrow width)
            - use_residual: bool (default True, essential for depth)
            - use_se_block: bool (default False, advanced channel attention)
            - se_ratio: int (default 16, reduction ratio for SE)
            - learning_rate: float (default 5e-4)
            - l2_reg: float (default 1e-4)
            - use_data_augmentation: bool
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Setup regularizer
    reg = None
    if params.get("l2_reg", 1e-4) > 0:
        reg = keras.regularizers.l2(params.get("l2_reg", 1e-4))
    
    # Calculate maximum possible layers based on input dimensions
    min_dim = min(input_shape[0], input_shape[1])
    max_layers = int(np.log2(min_dim)) if min_dim > 0 else 1
    
    # Pattern Scout: 6-7 layers is optimal for GAF
    requested_layers = params.get("num_conv_layers", 6)
    num_conv_layers = min(requested_layers, max_layers)
    
    if num_conv_layers < requested_layers:
        print(f"[GAF Optimized Deep CNN] Warning: Reduced layers from {requested_layers} to {num_conv_layers} due to input size constraints.")
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Data Augmentation (2D)
    if params.get("use_data_augmentation"):
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
    
    # Architecture parameters based on Pattern Scout findings
    base_filters = params.get("filters", 48)  # 48-64 is the sweet spot
    kernel_size = params.get("kernel_size", 7)  # 7x7 captures GAF temporal correlations better than 3x3 or 5x5
    use_constant_filters = params.get("use_constant_filters", True)  # Deep & Narrow strategy
    use_residual = params.get("use_residual", True)  # Essential for 6+ layers
    use_se = params.get("use_se_block", False)  # Advanced technique: channel attention
    
    for i in range(num_conv_layers):
        # Filter strategy: Constant (Deep & Narrow) vs Growth (Medium & Balanced)
        if use_constant_filters:
            num_filters = base_filters
        else:
            # Controlled growth: min(512, base * 2^i)
            num_filters = min(512, base_filters * (2**i))
        
        # Store shortcut for residual connection
        shortcut = x
        
        # Convolutional Block: Conv -> BN -> ReLU
        x = layers.Conv2D(
            num_filters,
            (kernel_size, kernel_size),
            padding="same",
            kernel_regularizer=reg,
            name=f"conv_block_{i+1}"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        
        # Advanced Technique: Squeeze-and-Excitation Block
        # Recalibrates channel-wise feature responses adaptively
        if use_se:
            se_ratio = params.get("se_ratio", 16)
            se_filters = max(1, num_filters // se_ratio)
            
            se = layers.GlobalAveragePooling2D()(x)
            se = layers.Reshape((1, 1, num_filters))(se)
            se = layers.Conv2D(se_filters, 1, activation="relu", padding="same")(se)
            se = layers.Conv2D(num_filters, 1, activation="sigmoid", padding="same")(se)
            x = layers.multiply([x, se], name=f"se_block_{i+1}")
        
        # Residual Connection (ResNet style)
        if use_residual:
            # Match dimensions if necessary
            if shortcut.shape[-1] != num_filters:
                shortcut = layers.Conv2D(
                    num_filters, 
                    (1, 1), 
                    padding="same",
                    kernel_regularizer=reg
                )(shortcut)
            
            # Add residual if spatial dimensions match (they should with 'same' padding)
            if shortcut.shape[1:3] == x.shape[1:3]:
                x = layers.Add()([x, shortcut])
        
        # MaxPooling (avoid after last conv layer to preserve spatial info for GAP)
        if i < num_conv_layers - 1:
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Global Average Pooling (reduces parameters vs Flatten, reduces overfitting)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense Layer: Pattern Scout recommends 256 (512 causes overfitting)
    dense_units = params.get("dense_units", 256)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_1"
    )(x)
    
    # Dropout: Pattern Scout recommends 0.45-0.50 for medium depth (5-6 layers)
    # For 7-8 layers, can use 0.30-0.40 due to implicit regularization from depth
    dropout_rate = params.get("dropout_rate", 0.5)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    
    # Build model
    model = keras.Model(inputs, outputs)
    
    # Optimizer: Pattern Scout recommends 5e-4 for medium depth configurations
    lr = params.get("learning_rate", 5e-4)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model