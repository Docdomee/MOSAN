def build_model(input_shape, num_classes, params: dict):
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np
    
    # Motif-optimized hyperparameters for 2D_GAF (69×69)
    # Depth: 7-8 convolutional layers (deep hierarchy for temporal correlations)
    num_conv_layers = params.get("num_conv_layers", 7)
    
    # Width: Fixed 128 filters (sweet spot, avoid >128 to prevent overfitting)
    num_filters = params.get("filters", 128)
    
    # Kernel: 5×5 receptive field (optimal efficiency-accuracy trade-off for 69×69)
    kernel_size = params.get("kernel_size", 5)
    
    # Regularization: Aggressive 0.5 dropout (mandatory for generalization)
    dropout_rate = params.get("dropout_rate", 0.5)
    
    # Classifier: High-capacity 512 dense units
    dense_units = params.get("dense_units", 512)
    
    # Optimization: Conservative learning rate ≤ 0.0001
    learning_rate = params.get("learning_rate", 1e-4)
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Data Augmentation (preserve capability from base model)
    if params.get("use_data_augmentation"):
        x = keras.layers.RandomFlip("horizontal")(x)
        x = keras.layers.RandomRotation(0.1)(x)
        x = keras.layers.RandomZoom(0.1)(x)
    
    # L2 Regularization
    reg = None
    if params.get("use_regularization", True):
        reg = keras.regularizers.l2(1e-4)
    
    # Deep & Narrow Convolutional Stack
    # Strategy: Pool every 2nd layer to allow 7-8 layers without spatial collapse
    for i in range(num_conv_layers):
        shortcut = x
        
        # Fixed-width convolution (128 filters, no exponential growth)
        x = keras.layers.Conv2D(
            num_filters,
            (kernel_size, kernel_size),
            padding="same",
            kernel_regularizer=reg
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Optional residual connections (preserve base model capability)
        if params.get("use_residual", False):
            if shortcut.shape[-1] != num_filters:
                shortcut = keras.layers.Conv2D(
                    num_filters, (1, 1), padding="same", kernel_regularizer=reg
                )(shortcut)
            x = keras.layers.Add()([x, shortcut])
        
        # Pooling strategy: Pool on odd indices (1, 3, 5...) and final layer
        # This maintains spatial resolution through the deep stack
        if (i % 2 == 1) or (i == num_conv_layers - 1):
            x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Global Average Pooling
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Aggressive dropout before high-capacity dense layer (Motif: dropout ≈ 0.5)
    x = keras.layers.Dropout(dropout_rate)(x)
    
    # High-capacity dense layer (512 units)
    x = keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=reg
    )(x)
    
    # Additional dropout for regularization (standard practice with high-capacity)
    x = keras.layers.Dropout(dropout_rate)(x)
    
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Conservative optimizer settings (lr ≤ 0.0001)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model