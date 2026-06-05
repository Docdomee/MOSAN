def build_model(input_shape, num_classes, params):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    
    # Use fixed optimal configuration based on Pattern Analysis
    # Core architecture: 4 conv layers with 256 filters, kernel size 3
    num_conv_layers = 4
    fixed_filters = 256
    fixed_kernel_size = 3
    
    # Input layer
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Regularization
    reg = keras.regularizers.l2(params.get("l2_reg", 1e-4))
    
    # Convolutional layers with residual connections
    for i in range(num_conv_layers):
        # Residual connection
        shortcut = x
        
        # Convolutional block
        x = keras.layers.Conv3D(
            filters=fixed_filters,
            kernel_size=fixed_kernel_size,
            padding="same",
            kernel_regularizer=reg
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Second conv for deeper representation
        x = keras.layers.Conv3D(
            filters=fixed_filters,
            kernel_size=fixed_kernel_size,
            padding="same",
            kernel_regularizer=reg
        )(x)
        x = keras.layers.BatchNormalization()(x)
        
        # Adjust shortcut for residual connection
        if shortcut.shape[-1] != fixed_filters:
            shortcut = keras.layers.Conv3D(
                filters=fixed_filters,
                kernel_size=1,
                padding="same",
                kernel_regularizer=reg
            )(shortcut)
        
        # Residual connection
        x = keras.layers.Add()([x, shortcut])
        x = keras.layers.Activation("relu")(x)
        
        # Pooling layer
        x = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # Global pooling and classifier
    x = keras.layers.GlobalAveragePooling3D()(x)
    x = keras.layers.Dense(
        params.get("dense_units", 128),
        activation="relu",
        kernel_regularizer=reg
    )(x)
    x = keras.layers.Dropout(params.get("dropout_rate", 0.3))(x)
    x = keras.layers.Dense(num_classes)(x)
    outputs = keras.layers.Activation("softmax", dtype="float32")(x)
    
    # Model compilation
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(
        learning_rate=params.get("learning_rate", 0.0005)
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model