import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, params):
    """
    Enhanced ResNet-1D Architecture with True Residual Connections, Improved Regularization,
    and Modern Training Strategies.

    Key Enhancements:
    - True ResNet-style residual blocks with projection shortcuts
    - Consistent BatchNorm usage for stability
    - Spatial dropout for structured regularization
    - Global Average Pooling instead of Flatten for better generalization
    - Weight decay as additional regularization
    - Configurable depth_scaling for modularity
    """

    # --- Default Parameters from Design Strategy ---
    filters = params.get("filters", 128)
    kernel_size = params.get("kernel_size", 3)
    dense_units = params.get("dense_units", 512)
    dropout_rate = params.get("dropout_rate", 0.3)
    learning_rate = params.get("learning_rate", 0.0005)
    batch_norm = params.get("batch_norm", True)
    weight_decay = params.get("weight_decay", 1e-5)
    depth_scaling = params.get("depth_scaling", 2) # Multiplier on residual blocks per stage
    
    # --- Input ---
    inputs = keras.Input(shape=input_shape)

    # --- Initial Convolutional Block ---
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding='same',
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        bias_regularizer=keras.regularizers.l2(weight_decay)
    )(inputs)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # --- Residual Block Definition ---
    def residual_block(x, filters, stride=1, dropout_rate=0.0, stage='_', block='_'):
        """A standard bottleneck residual block adapted for 1D."""
        shortcut = x
        f1, f2, f3 = filters
        
        # First component of the main path
        x = layers.Conv1D(
            filters=f1, kernel_size=1, strides=stride, padding='valid',
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            bias_regularizer=keras.regularizers.l2(weight_decay),
            name=f'res{stage}_{block}_branch2a_conv'
        )(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f'res{stage}_{block}_branch2a_bn')(x)
        x = layers.Activation('relu', name=f'res{stage}_{block}_branch2a_relu')(x)

        # Second component of the main path
        x = layers.Conv1D(
            filters=f2, kernel_size=kernel_size, strides=1, padding='same',
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            bias_regularizer=keras.regularizers.l2(weight_decay),
            name=f'res{stage}_{block}_branch2b_conv'
        )(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f'res{stage}_{block}_branch2b_bn')(x)
        x = layers.Activation('relu', name=f'res{stage}_{block}_branch2b_relu')(x)
        
        # Third component of the main path
        x = layers.Conv1D(
            filters=f3, kernel_size=1, strides=1, padding='valid',
            kernel_regularizer=keras.regularizers.l2(weight_decay),
            bias_regularizer=keras.regularizers.l2(weight_decay),
            name=f'res{stage}_{block}_branch2c_conv'
        )(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f'res{stage}_{block}_branch2c_bn')(x)

        # Residual Connection
        if stride != 1 or shortcut.shape[-1] != f3:
            shortcut = layers.Conv1D(
                filters=f3, kernel_size=1, strides=stride, padding='valid',
                kernel_regularizer=keras.regularizers.l2(weight_decay),
                bias_regularizer=keras.regularizers.l2(weight_decay),
                name=f'res{stage}_{block}_branch1_conv'
            )(shortcut)
            if batch_norm:
                shortcut = layers.BatchNormalization(name=f'res{stage}_{block}_branch1_bn')(shortcut)

        x = layers.Add(name=f'res{stage}_{block}_add')([x, shortcut])
        x = layers.Activation('relu', name=f'res{stage}_{block}_out')(x)
        
        # Optional Structured Dropout After Residual Block
        if dropout_rate > 0:
            x = layers.SpatialDropout1D(rate=dropout_rate)(x)
            
        return x

    # --- Stage 1 ---
    stage_filters = [filters//4, filters//2, filters]
    for i in range(depth_scaling):
        x = residual_block(x, stage_filters, stride=1, dropout_rate=dropout_rate/2, stage='2', block=str(i+1))

    # --- Stage 2 (Downsampling via Strided Conv) ---
    stage_filters = [filters, filters*2, filters*4]
    for i in range(depth_scaling):
        stride = 2 if i == 0 else 1
        x = residual_block(x, stage_filters, stride=stride, dropout_rate=dropout_rate, stage='3', block=str(i+1))

    # --- Stage 3 (Optional Deeper Stack) ---
    stage_filters = [filters*2, filters*4, filters*8]
    for i in range(depth_scaling):
        stride = 2 if i == 0 else 1
        x = residual_block(x, stage_filters, stride=stride, dropout_rate=dropout_rate, stage='4', block=str(i+1))

    # --- Global Average Pooling instead of Flatten ---
    x = layers.GlobalAveragePooling1D()(x)

    # --- Fully Connected Layers ---
    x = layers.Dense(
        units=dense_units,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        bias_regularizer=keras.regularizers.l2(weight_decay)
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output layer
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        kernel_regularizer=keras.regularizers.l2(weight_decay),
        bias_regularizer=keras.regularizers.l2(weight_decay)
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Enhanced_ResNet1D")

    # --- Optimizer with Weight Decay Handling ---
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model