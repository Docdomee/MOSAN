def build_model(input_shape, num_classes, params):
    """
    Improved 2D GAF CNN Architecture based on pattern analysis.
    
    Optimizations:
    - Deeper architecture (5-8 conv layers)
    - Smaller kernel sizes (3x3)
    - Residual connections for better gradient flow
    - Batch normalization for training stability
    - Moderate dropout (0.25-0.4)
    - Appropriate filter progression
    """
    import tensorflow as tf
    from tensorflow import keras
    import numpy as np

    # --- Parameter Configuration ---
    # Default values aligned with pattern analysis
    num_conv_layers = params.get("num_conv_layers", 6)  # 5-8 range
    base_filters = params.get("filters", 64)  # Start with 64
    kernel_size = params.get("kernel_size", 3)  # 3x3 kernels
    dropout_rate = params.get("dropout_rate", 0.3)  # Moderate dropout
    dense_units = params.get("dense_units", 256)  # 256 units
    learning_rate = params.get("learning_rate", 5e-4)  # Within 1e-4 to 1e-3
    use_residual = params.get("use_residual", True)  # Enable residual connections
    reg_strength = params.get("l2_reg", 1e-4)  # L2 regularization
    
    # Ensure we have a valid number of layers
    num_conv_layers = max(3, min(8, num_conv_layers))  # Clamp between 3-8
    
    # --- Model Definition ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # --- Data Augmentation (Optional) ---
    if params.get("use_data_augmentation", False):
        x = keras.layers.RandomFlip("horizontal")(x)
        x = keras.layers.RandomRotation(0.1)(x)
        x = keras.layers.RandomZoom(0.1)(x)
    
    # Regularizer
    reg = keras.regularizers.l2(reg_strength)
    
    # --- Convolutional Layers ---
    for i in range(num_conv_layers):
        # Filter progression: double every 2 layers
        num_filters = base_filters * (2 ** (i // 2))
        num_filters = min(512, num_filters)  # Cap at 512 filters
        
        # Residual shortcut
        shortcut = x
        
        # First conv layer
        x = keras.layers.Conv2D(
            num_filters, 
            (kernel_size, kernel_size), 
            padding="same",
            kernel_regularizer=reg,
            name=f"conv_{i}_1"
        )(x)
        x = keras.layers.BatchNormalization(name=f"bn_{i}_1")(x)
        x = keras.layers.Activation("relu", name=f"relu_{i}_1")(x)
        
        # Second conv layer
        x = keras.layers.Conv2D(
            num_filters, 
            (kernel_size, kernel_size), 
            padding="same",
            kernel_regularizer=reg,
            name=f"conv_{i}_2"
        )(x)
        x = keras.layers.BatchNormalization(name=f"bn_{i}_2")(x)
        
        # Residual connection
        if use_residual:
            # Adjust shortcut if needed
            if shortcut.shape[-1] != num_filters:
                shortcut = keras.layers.Conv2D(
                    num_filters, 
                    (1, 1), 
                    padding="same",
                    name=f"shortcut_conv_{i}"
                )(shortcut)
                shortcut = keras.layers.BatchNormalization(name=f"shortcut_bn_{i}")(shortcut)
            
            x = keras.layers.Add(name=f"add_{i}")([x, shortcut])
        
        x = keras.layers.Activation("relu", name=f"relu_{i}_2")(x)
        
        # Pooling
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), name=f"pool_{i}")(x)
        
        # Add dropout after every 2 layers
        if i % 2 == 1 and dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)
    
    # --- Global Average Pooling ---
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    # --- Dense Layers ---
    x = keras.layers.Dense(
        dense_units, 
        activation="relu",
        kernel_regularizer=reg,
        name="dense"
    )(x)
    
    # Final dropout
    if dropout_rate > 0:
        x = keras.layers.Dropout(dropout_rate, name="final_dropout")(x)
    
    # --- Output Layer ---
    outputs = keras.layers.Dense(
        num_classes, 
        activation="softmax", 
        dtype="float32",
        name="output"
    )(x)
    
    # --- Model Compilation ---
    model = keras.Model(inputs, outputs, name="Improved_2D_GAF_CNN")
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model