import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, params):
    """
    Deep Residual 2D CNN for GAF Images from SERS Spectra.
    
    Hypothesis: Switching from 1D CNN on raw spectra to 2D CNN on GAF images
    will escape the performance plateau by providing a more structured representation
    that captures temporal/sequential relationships in the spectral data.
    
    Design Choices Based on Pattern Scout Analysis:
    - Deep architecture with residual connections (20+ conv layers)
    - Moderate filter growth (64->512)
    - Aggressive dropout (0.5+ in dense layers)
    - Batch normalization after every conv layer
    - Conservative learning rate range
    - Global Average Pooling before classifier
    - Larger kernel sizes (5x5) to capture broader patterns in GAF
    """
    
    # Defensive cast
    input_shape = tuple(input_shape)
    
    # --- Hyperparameters from params or defaults ---
    dropout_rate = params.get("dropout_rate", 0.55)
    learning_rate = params.get("learning_rate", 1e-4)
    l2_reg = params.get("l2_reg", 1e-4)
    use_residual = params.get("use_residual", True)
    filters_base = params.get("filters_base", 64)
    kernel_size = params.get("kernel_size", 5)  # Scout contradiction - using larger kernels
    weight_initializer = params.get("weight_initializer", "he_normal")
    
    # Input layer
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Initial convolution block
    x = layers.Conv2D(filters_base, kernel_size=kernel_size, strides=2, padding='same',
                      kernel_initializer=weight_initializer,
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Residual block configuration
    def residual_block(x, filters, stride=1, conv_shortcut=False):
        """Standard bottleneck residual block"""
        shortcut = x
        if conv_shortcut:
            shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                   kernel_initializer=weight_initializer,
                                   kernel_regularizer=keras.regularizers.l2(l2_reg))(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # 1x1 conv
        x = layers.Conv2D(filters, 1, strides=1,
                         kernel_initializer=weight_initializer,
                         kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # 3x3 conv (using our specified kernel size)
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                         kernel_initializer=weight_initializer,
                         kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # 1x1 conv expansion
        x = layers.Conv2D(4 * filters, 1, strides=1,
                         kernel_initializer=weight_initializer,
                         kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = layers.BatchNormalization()(x)

        # Add shortcut
        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        return x

    # --- Residual Block Stack ---
    # Stage 1
    x = residual_block(x, filters_base, stride=1, conv_shortcut=True)
    x = residual_block(x, filters_base, stride=1)
    x = residual_block(x, filters_base, stride=1)

    # Stage 2
    x = residual_block(x, filters_base * 2, stride=2, conv_shortcut=True)
    x = residual_block(x, filters_base * 2, stride=1)
    x = residual_block(x, filters_base * 2, stride=1)
    x = residual_block(x, filters_base * 2, stride=1)

    # Stage 3
    x = residual_block(x, filters_base * 4, stride=2, conv_shortcut=True)
    x = residual_block(x, filters_base * 4, stride=1)
    x = residual_block(x, filters_base * 4, stride=1)
    x = residual_block(x, filters_base * 4, stride=1)
    x = residual_block(x, filters_base * 4, stride=1)
    x = residual_block(x, filters_base * 4, stride=1)

    # Stage 4
    x = residual_block(x, filters_base * 8, stride=2, conv_shortcut=True)
    x = residual_block(x, filters_base * 8, stride=1)
    x = residual_block(x, filters_base * 8, stride=1)

    # Global average pooling instead of flatten
    x = layers.GlobalAveragePooling2D()(x)
    
    # Aggressive dropout before dense layer
    x = layers.Dropout(dropout_rate)(x)
    
    # Dense classification layer with L2 regularization
    outputs = layers.Dense(
        num_classes, 
        activation='softmax',
        kernel_initializer=weight_initializer,
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)

    # Create model
    model = keras.Model(inputs, outputs, name="DeepGAFResNet2D")
    
    # Compile with conservative learning rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model