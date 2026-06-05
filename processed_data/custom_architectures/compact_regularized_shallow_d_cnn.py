def build_model(input_shape, num_classes, params: dict):
    """
    Compact Regularized Shallow 1D CNN Architecture.
    
    Evolution of base 1D_CNN incorporating Pattern Analysis findings:
    - Exactly 2 conv layers (depth penalty for >2)
    - Fixed 32-64 filters per layer (no expansion, capacity sweet spot)
    - Kernel size 5 (optimal for sequence length ~55)
    - Aggressive dropout 0.85-0.90 between conv layers (anti-pattern: <0.8)
    - Minimal dense head (32 units) with L2 regularization
    - Learning rate ~7e-5 for stable convergence
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    
    # --- Architectural Constraints (Pattern Analysis Enforcement) ---
    # Force shallow depth: 2 layers is the optimal sweet spot
    requested_layers = params.get("num_conv_layers", 2)
    if requested_layers != 2:
        print(f"[Architect Warning] Requested {requested_layers} layers deviates from optimal depth of 2. "
              f"Forcing 2 layers to prevent overfitting (depth penalty observed).")
    num_conv_layers = 2
    
    # Capacity constraint: 32-64 filters optimal, strictly enforce ceiling at 64
    base_filters = params.get("filters", 32)
    if base_filters > 64:
        print(f"[Architect Warning] Filters={base_filters} exceeds safe capacity (max 64). "
              f"Capping at 64 to prevent catastrophic overfitting.")
        base_filters = 64
    elif base_filters < 32:
        base_filters = 32
    
    # Receptive field constraint: kernel 5 optimal for length 55 sequences
    kernel_size = params.get("kernel_size", 5)
    if kernel_size > 5:
        print(f"[Architect Warning] Kernel size {kernel_size} exceeds optimal size 5 for short sequences.")
    
    # Aggressive regularization: dropout must be >0.81, preferably 0.85-0.90
    dropout_conv = params.get("dropout_rate", 0.88)
    if dropout_conv < 0.8:
        print(f"[Architect Warning] Dropout {dropout_conv} insufficient for this data. "
              f"Increasing to 0.88 to satisfy regularization requirements.")
        dropout_conv = 0.88
    
    # Dense layer minimization: 32 units generalizes better than 128
    dense_units = params.get("dense_units", 32)
    if dense_units > 64:
        print(f"[Architect Warning] Dense units {dense_units} too large (overfitting vector). Capping at 64.")
        dense_units = 64
    
    dropout_dense = params.get("dropout_dense", 0.5)
    
    # L2 weight decay alongside high dropout (Next Steps recommendation)
    l2_lambda = params.get("l2_lambda", 1e-4)
    reg = keras.regularizers.l2(l2_lambda)
    
    # Training dynamics: LR ~7e-5 showed stable convergence
    learning_rate = params.get("learning_rate", 7e-5)
    
    # --- Model Construction ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Data Augmentation (preserved from base)
    if params.get("use_data_augmentation", False):
        x = keras.layers.GaussianNoise(0.1)(x)
    
    # Convolutional Block 1
    x = keras.layers.Conv1D(
        base_filters,
        kernel_size,
        padding="same",
        kernel_regularizer=reg,
        kernel_initializer="he_normal"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(dropout_conv)(x)  # Aggressive regularization between layers
    x = keras.layers.MaxPooling1D(2)(x)
    
    # Convolutional Block 2 (fixed width, no expansion)
    x = keras.layers.Conv1D(
        base_filters,
        kernel_size,
        padding="same",
        kernel_regularizer=reg,
        kernel_initializer="he_normal"
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(dropout_conv)(x)  # Aggressive regularization between layers
    x = keras.layers.MaxPooling1D(2)(x)
    
    # Global Average Pooling (preserved from base)
    x = keras.layers.GlobalAveragePooling1D()(x)
    
    # Compact Classifier Head (32 units)
    x = keras.layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=reg,
        kernel_initializer="he_normal"
    )(x)
    x = keras.layers.Dropout(dropout_dense)(x)
    
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model