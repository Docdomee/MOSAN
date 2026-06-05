import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, params: dict):
    """
    Regularized Capacity Volumetric CNN.
    
    Fixed: Added dynamic dimension checking to prevent Negative Dimension 
    errors during MaxPooling3D when input shapes are small.
    """
    
    # --- Hyperparameter Extraction ---
    num_conv_layers = params.get("num_conv_layers", 5)
    base_filters = params.get("filters", 64)
    kernel_size = params.get("kernel_size", 5)
    dense_units = params.get("dense_units", 512)
    dropout_rate = params.get("dropout_rate", 0.5) 
    use_residual = params.get("use_residual", True)
    
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # --- Volumetric Convolutional Backbone ---
    for i in range(num_conv_layers):
        filters = base_filters * (2**i if i < 3 else 8) 
        
        shortcut = x
        
        # Convolution Block
        x = layers.Conv3D(
            filters=filters, 
            kernel_size=kernel_size, 
            padding='same', 
            kernel_initializer='he_normal'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # --- Safe Max Pooling ---
        # Check if any dimension is too small to be pooled (size <= 1)
        # input_shape is (D, H, W). We check current tensor shape.
        current_shape = x.shape
        # We only pool if all spatial dimensions are > 1
        # Keras shapes can be None or Dimension objects; convert to int
        can_pool = True
        for dim in current_shape[1:4]: # Depth, Height, Width
            if dim is not None and dim <= 1:
                can_pool = False
                break
        
        if can_pool:
            x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
        
        # Residual Connection
        if use_residual:
            # Match filters
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv3D(filters, (1, 1, 1), padding='same')(shortcut)
            
            # Match spatial dimensions (only pool shortcut if main path was pooled)
            if can_pool:
                shortcut = layers.MaxPooling3D(pool_size=(2, 2, 2))(shortcut)
            
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)

    # --- Global Representation ---
    x = layers.GlobalAveragePooling3D()(x)
    
    # --- Robustness Head ---
    x = layers.Dense(dense_units, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model