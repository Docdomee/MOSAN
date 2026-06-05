import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, params: dict):
    """
    Hierarchical Residual 1D CNN.
    
    Improvements implemented per Design Strategy:
    1. Width over Depth: Increased base filters to capture complex patterns.
    2. Progressive Downsampling: Conv -> Pool blocks for hierarchical feature extraction.
    3. Residual Learning: Full implementation of skip connections to prevent vanishing gradients.
    4. Regularization: L2 weight decay and Dropout to bridge the train/val gap.
    5. Activation: LeakyReLU to prevent dead neurons.
    """
    
    # --- Hyperparameters from params or Design Strategy defaults ---
    # Shift from Depth to Width: Using higher filter counts (128/256)
    base_filters = params.get("filters", 128) 
    num_residual_blocks = params.get("num_conv_layers", 3) # Interpreted as blocks of residuals
    dropout_rate = params.get("dropout_rate", 0.4)
    weight_decay = params.get("weight_decay", 1e-4)
    
    # Regularizer
    l2_reg = keras.regularizers.l2(weight_decay)
    
    def residual_block(x, filters, kernel_size=3):
        """
        A standard residual block: Conv1D -> BN -> LeakyReLU -> Conv1D -> BN -> Add -> LeakyReLU
        """
        shortcut = x
        
        # First Convolution
        x = layers.Conv1D(
            filters, kernel_size, padding='same', 
            kernel_regularizer=l2_reg, kernel_initializer='he_normal'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        
        # Second Convolution
        x = layers.Conv1D(
            filters, kernel_size, padding='same', 
            kernel_regularizer=l2_reg, kernel_initializer='he_normal'
        )(x)
        x = layers.BatchNormalization()(x)
        
        # Projection shortcut if dimensions change
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, padding='same', kernel_regularizer=l2_reg)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
            
        x = layers.Add()([x, shortcut])
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    # --- Model Architecture ---
    inputs = keras.Input(shape=input_shape)
    
    # Initial Feature Extraction
    x = layers.Conv1D(
        base_filters, 7, strides=2, padding='same', 
        kernel_regularizer=l2_reg, kernel_initializer='he_normal'
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    # Progressive Downsampling Motif
    # We create stages: each stage has residual blocks followed by a MaxPooling layer
    current_filters = base_filters
    for i in range(num_residual_blocks):
        # Increase width as we go deeper (diversity)
        current_filters = min(current_filters * 2, 512) 
        
        # Residual learning block
        x = residual_block(x, current_filters)
        
        # Downsample to create hierarchical representation
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(dropout_rate)(x)

    # Final Aggregation
    # Global Average Pooling instead of Flatten to reduce parameter count and overfitting
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification Head
    x = layers.Dense(64, activation='leaky_relu', kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Note: Learning rate (1e-4) should be handled by the optimizer in the training loop
    return model