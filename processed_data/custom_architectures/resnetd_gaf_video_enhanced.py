def build_model(input_shape, num_classes, params):
    """
    Enhanced ResNetD-style model for 3D GAF Video inputs.

    Innovations Implemented:
    - Moderate kernel sizes (3x3x3) with progressive filters [32→64→128]
    - Shallow to moderate depth (3 Conv3D blocks with residual connections)
    - Spatial+Temporal BatchNormalization after each conv block
    - SpatialDropout3D for better regularization in volumetric data
    - Final dense layer with high dropout (0.5) to prevent overfitting
    - L2 Regularization on final dense layer
    - Residual path matching via 1x1x1 conv when needed
    """

    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    # --- Hyperparameters ---
    dropout_rate_dense = params.get("dropout_rate_dense", 0.5)
    l2_reg_dense = params.get("l2_reg_dense", 1e-4)
    use_residual = params.get("use_residual", True)
    initial_filters = params.get("initial_filters", 32)

    # --- Model Definition ---
    inputs = layers.Input(shape=input_shape)
    x = inputs

    def conv_block(x, filters, kernel_size=(3, 3, 3), use_residual=True):
        """Convolutional block with optional residual connection."""
        shortcut = x

        # First conv layer
        x = layers.Conv3D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Second conv layer
        x = layers.Conv3D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)

        # Adjust shortcut for residual connection if needed
        if use_residual:
            if shortcut.shape[-1] != x.shape[-1]:
                shortcut = layers.Conv3D(filters, (1, 1, 1), padding='same', kernel_initializer='he_normal')(shortcut)
            x = layers.Add()([x, shortcut])

        x = layers.Activation('relu')(x)
        return x

    # --- Convolutional Backbone ---
    # Block 1: Use (1, 2, 2) pooling to preserve temporal dimension
    x = conv_block(x, initial_filters, use_residual=use_residual)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)  # Changed from (2, 2, 2) to (1, 2, 2)
    x = layers.SpatialDropout3D(0.3)(x)

    # Block 2: Use (1, 2, 2) pooling to preserve temporal dimension
    x = conv_block(x, initial_filters * 2, use_residual=use_residual)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)  # Changed from (2, 2, 2) to (1, 2, 2)
    x = layers.SpatialDropout3D(0.4)(x)

    # Block 3: Use (1, 2, 2) pooling to preserve temporal dimension
    x = conv_block(x, initial_filters * 4, use_residual=use_residual)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)  # Changed from (2, 2, 2) to (1, 2, 2)
    x = layers.SpatialDropout3D(0.5)(x)

    # --- Classification Head ---
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg_dense)
    )(x)
    x = layers.Dropout(dropout_rate_dense)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model