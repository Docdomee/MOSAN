import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, params: dict):
    """
    Deep and Narrow Volumetric CWT ResNet.
    
    Architectural Strategy:
    1. Deep and Narrow: 6 layers with only 32 filters to prevent overfitting.
    2. Large Receptive Field: Kernel size 7 to capture volumetric CWT patterns.
    3. Parameter Control: Dense units capped at 256.
    4. Generalization: L2 regularization and Residual connections to bridge the shifted test gap.
    5. Dimension Guard: Conditional pooling to prevent negative dimension errors on small inputs.
    """
    
    # --- Hyperparameters from Design Strategy ---
    num_conv_layers = params.get("num_conv_layers", 6)
    filters = params.get("filters", 32)
    kernel_size = params.get("kernel_size", 7)
    dense_units = params.get("dense_units", 256)
    dropout_rate = params.get("dropout_rate", 0.15) 
    use_residual = params.get("use_residual", True)
    l2_reg_val = params.get("l2_reg", 1e-4) if params.get("use_l2_regularization", True) else 0.0
    
    reg = keras.regularizers.l2(l2_reg_val)
    
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # --- Convolutional Backbone ---
    for i in range(num_conv_layers):
        # Store identity for residual connection
        identity = x
        
        # Convolutional Layer
        x = layers.Conv3D(
            filters=filters, 
            kernel_size=kernel_size, 
            padding='same', 
            kernel_regularizer=reg,
            activation='relu'
        )(x)
        
        x = layers.BatchNormalization()(x)
        
        # --- Dimension Guard for Pooling ---
        # We check the current tensor shape to avoid pooling if dimensions are too small
        # input_shape is (D, H, W, C). We check if min(D, H, W) >= 2
        current_shape = x.shape
        # Handle both TensorShape and tuple formats
        spatial_dims = current_shape[1:4] 
        
        pooled = False
        if all(d is not None and d >= 2 for d in spatial_dims):
            x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
            pooled = True
        
        # Residual Connection
        if use_residual:
            # 1. Handle Spatial Downsampling: 
            # Only pool identity if the main path was actually pooled
            if pooled:
                identity = layers.MaxPooling3D(pool_size=(2, 2, 2))(identity)
            
            # 2. Handle Channel Alignment:
            # Project identity to match the number of filters in x
            if identity.shape[-1] != filters:
                identity = layers.Conv3D(filters, (1, 1, 1), padding='same')(identity)
            
            x = layers.Add()([x, identity])
            x = layers.Activation('relu')(x)

    # --- Classification Head ---
    x = layers.GlobalAveragePooling3D()(x)
    
    x = layers.Dense(
        dense_units, 
        activation='relu', 
        kernel_regularizer=reg
    )(x)
    
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model