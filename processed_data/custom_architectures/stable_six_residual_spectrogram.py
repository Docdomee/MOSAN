#CREATE_NEW_MANIFEST
"""
<representation>2D_SPECTROGRAM</representation>
"""

def build_model(input_shape, num_classes, params):
    """
    Stable Six Residual 2D Spectrogram CNN.
    
    Implements the "Stable Six" motif (6 conv layers, 128 filters, kernel 3) 
    with residual connections as the advanced technique for improved gradient 
    flow and potential depth scaling. Optimized for spectrogram classification
    with strict adherence to the 4.1M parameter Pareto frontier.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras import backend as K
    
    # --- Stable Six Configuration (Locked Backbone) ---
    # Architecture constants derived from successful motif analysis
    NUM_CONV_LAYERS = 6          # Stability point for spectrogram depth
    NUM_FILTERS = 128            # Sweet spot for feature capacity (4.1M param target)
    KERNEL_SIZE = 3              # Critical for local spectral coherence (avoid 5)
    DENSE_UNITS = 512            # Classifier capacity
    DROPOUT_RATE = 0.5           # Exactly 0.5 - precision critical
    INITIAL_LR = 1e-5            # Sharp performance cliff above 5e-5
    
    # Advanced technique flags
    use_residual = params.get("use_residual", True)       # Enable skip connections
    use_batchnorm = params.get("use_batchnorm", False)    # Avoid unless overfitting
    use_augmentation = params.get("use_data_augmentation", False)
    
    # Regularization
    reg = keras.regularizers.l2(1e-4) if params.get("use_regularization", True) else None
    
    inputs = layers.Input(shape=input_shape, name="spectrogram_input")
    x = inputs
    
    # --- Data Augmentation (2D Spectrogram Safe) ---
    if use_augmentation:
        x = layers.RandomFlip("horizontal")(x)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
    
    # --- Convolutional Backbone (The "Stable Six") ---
    for i in range(NUM_CONV_LAYERS):
        # Residual shortcut for gradient highway (pre-activation capture)
        shortcut = x
        
        # Main convolutional path
        x = layers.Conv2D(
            filters=NUM_FILTERS,
            kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
            strides=(1, 1),
            padding="same",
            kernel_regularizer=reg,
            name=f"conv_block_{i+1}"
        )(x)
        
        # BatchNorm only if explicitly requested (anti-pattern in base motif)
        if use_batchnorm:
            x = layers.BatchNormalization()(x)
        
        x = layers.Activation("relu")(x)
        
        # Residual connection implementation
        if use_residual:
            # Robust channel dimension extraction using backend.int_shape
            shortcut_shape = K.int_shape(shortcut)
            shortcut_channels = shortcut_shape[-1] if shortcut_shape else None
            
            # Match dimensions if necessary (channel mismatch handling)
            if shortcut_channels != NUM_FILTERS:
                shortcut = layers.Conv2D(
                    NUM_FILTERS, 
                    kernel_size=(1, 1), 
                    strides=(1, 1),
                    padding="same",
                    kernel_regularizer=reg,
                    name=f"projection_{i+1}"
                )(shortcut)
            
            # Skip connection with ReLU activation after addition (ResNet v1 style)
            x = layers.Add(name=f"skip_add_{i+1}")([x, shortcut])
            x = layers.Activation("relu")(x)
        
        # Spatial downsampling (2x reduction per layer)
        # After 6 layers: 64x spatial reduction (e.g., 512x512 -> 8x8)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"pool_{i+1}")(x)
    
    # --- Classifier Head ---
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    
    x = layers.Dense(
        DENSE_UNITS,
        activation="relu",
        kernel_regularizer=reg,
        name="dense_512"
    )(x)
    
    # Mandatory Dropout 0.5 (anti-pattern to deviate)
    x = layers.Dropout(DROPOUT_RATE)(x)
    
    outputs = layers.Dense(
        num_classes, 
        activation="softmax",
        name="predictions"
    )(x)
    
    # --- Optimization Configuration ---
    # Cosine decay for 300 epochs as per design strategy
    decay_epochs = params.get("epochs", 300)
    steps_per_epoch = params.get("steps_per_epoch", 100)
    decay_steps = decay_epochs * steps_per_epoch
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=params.get("learning_rate", INITIAL_LR),
        decay_steps=decay_steps,
        alpha=0.0  # Decay to 0
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model