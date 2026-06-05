import numpy as np
import tensorflow as tf
from tensorflow import keras

def build_model(input_shape, num_classes, params: dict):
    """
    Advanced 3D CNN for 3D_VIDEO.
    
    FIX:
    - Resolves 'arith.mulf' XLA mismatch by removing conflicting explicit dtype overrides.
    - In mixed-precision environments, forcing dtypes on specific layers while the 
      global policy is float16 often causes XLA fusion crashes (Triton).
    - We rely on the global mixed_precision policy for internal layers and only 
      ensure the final output is float32 for softmax stability.
    """
    
    # Internal helper for regularization
    def _get_regularizer(params):
        l2_val = params.get("l2_reg", 1e-4)
        return keras.regularizers.l2(l2_val)

    # Defensive dimension handling
    input_shape = tuple(input_shape)
    # input_shape is (D, H, W, C)
    min_spatial_dim = min(input_shape[1], input_shape[2])
    
    # Stabilization Phase: Ensure we don't exceed dimensions
    max_layers = int(np.floor(np.log2(min_spatial_dim))) if min_spatial_dim > 0 else 1
    requested_layers = params.get("num_conv_layers", 3)
    num_conv_layers = min(requested_layers, max_layers)

    if num_conv_layers < requested_layers:
        print(f"[Architect Warning] Data dimensions limit layers to {num_conv_layers}.")

    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # Cast input to float32. The mixed_precision policy will handle 
    # the conversion to float16 for the first Conv3D layer automatically.
    x = tf.cast(inputs, tf.float32)
    
    reg = _get_regularizer(params)

    # Volumetric Feature Extraction Pipeline
    for i in range(num_conv_layers):
        num_filters = min(512, params.get("filters", 32) * (2**i))
        dilation_rate = 1 if i == 0 else (2**i if params.get("use_dilation", True) else 1)
        
        shortcut = x
        
        # 3D Convolutional Block
        x = keras.layers.Conv3D(
            filters=num_filters, 
            kernel_size=params.get("kernel_size", 3), 
            padding="same", 
            dilation_rate=dilation_rate,
            kernel_regularizer=reg
        )(x)
        
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        
        # Residual Connection
        if params.get("use_residual", True):
            if shortcut.shape[-1] != num_filters:
                shortcut = keras.layers.Conv3D(
                    num_filters, 
                    kernel_size=1, 
                    padding="same"
                )(shortcut)
            shortcut = keras.layers.BatchNormalization()(shortcut)
            x = keras.layers.Add()([x, shortcut])
            
        # Downsampling: Pool spatial dimensions (H, W), preserve temporal/depth (D)
        if x.shape[1] is not None and x.shape[1] > 1 and x.shape[2] is not None and x.shape[2] > 1:
            x = keras.layers.MaxPooling3D(pool_size=(1, 2, 2))(x)

    # Temporal/Volumetric Bottleneck
    x = keras.layers.GlobalAveragePooling3D()(x)
    
    # Dense head
    x = keras.layers.Dense(
        params.get("dense_units", 128), 
        activation="relu", 
        kernel_regularizer=reg
    )(x)
    
    x = keras.layers.Dropout(params.get("dropout_rate", 0.3))(x)
    
    # Output Layer
    # We use a separate Dense layer for the logits and a separate Activation 
    # to ensure we can cast the final result to float32 for softmax stability.
    logits = keras.layers.Dense(num_classes)(x)
    
    # Explicitly cast logits to float32 before softmax to prevent XLA type mismatch 
    # and numerical instability.
    logits_f32 = tf.cast(logits, tf.float32)
    outputs = keras.layers.Activation("softmax")(logits_f32)
    
    model = keras.Model(inputs, outputs)
    
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3))
    
    # We set jit_compile=False here as a preference, but since the worker overrides it,
    # the key fix is the removal of conflicting dtypes in the layers above.
    model.compile(
        optimizer=optimizer, 
        loss="categorical_crossentropy", 
        metrics=["accuracy"],
        jit_compile=False 
    )
    
    return model