def build_model(input_shape, num_classes, params: dict):
    from tensorflow.keras import layers, models, regularizers, backend as K
    import numpy as np
    
    # Extract params with defaults
    num_residual_blocks = params.get('num_residual_blocks', 5)
    filters = params.get('filters', 64)
    kernel_size = params.get('kernel_size', 3)
    attention_heads = params.get('attention_heads', 4)
    dropout_rate = params.get('dropout_rate', 0.3)
    entropy_lambda = params.get('entropy_lambda', 0.01)  # For entropy regularization
    
    # Custom entropy regularization layer (approximates entropy stabilization)
    class EntropyRegularizer(layers.Layer):
        def __init__(self, lambda_reg=0.01, **kwargs):
            super().__init__(**kwargs)
            self.lambda_reg = lambda_reg
        
        def call(self, inputs):
            # Compute entropy of activations (simplified as negative log of mean abs values)
            entropy = -K.mean(K.log(K.abs(inputs) + 1e-8))
            self.add_loss(self.lambda_reg * entropy)
            return inputs
    
    # Input layer (assuming input_shape is for 2D, e.g., (height, width, channels))
    inputs = layers.Input(shape=input_shape)
    
    # Initial conv block
    x = layers.Conv2D(filters, (kernel_size, kernel_size), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Residual blocks with attention
    for _ in range(num_residual_blocks):
        # Residual connection
        res = x
        
        # Conv block
        x = layers.Conv2D(filters, (kernel_size, kernel_size), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Attention mechanism (adapted for 2D by reshaping to sequence-like for attention)
        shape_before = K.int_shape(x)  # (batch, height, width, filters)
        h, w = shape_before[1], shape_before[2]
        seq_len = h * w
        
        # Reshape to (batch, seq_len, filters) for attention
        x_reshaped = layers.Reshape((seq_len, filters))(x)
        
        # Multi-head attention (using dense layers as proxy)
        attn_outputs = []
        for head in range(attention_heads):
            query = layers.Dense(filters // attention_heads)(x_reshaped)
            key = layers.Dense(filters // attention_heads)(x_reshaped)
            value = layers.Dense(filters // attention_heads)(x_reshaped)
            attn = layers.Attention()([query, value, key])
            attn_outputs.append(attn)
        
        attn_concat = layers.Concatenate()(attn_outputs)
        attn_out = layers.Dense(filters)(attn_concat)
        
        # Reshape back to (batch, height, width, filters)
        attn_out = layers.Reshape((h, w, filters))(attn_out)
        
        # Add residual and entropy regularizer
        x = layers.Add()([attn_out, res])
        x = EntropyRegularizer(entropy_lambda)(x)
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model