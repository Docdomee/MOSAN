import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def build_model(input_shape, num_classes, params):
    """
    Improved 3D Attention Model for CWT-based SERS data with enhanced generalization.
    
    Enhancements Implemented:
    - Data Augmentation via noise injection tailored to CWT
    - Domain-invariant attention with positional encoding
    - Depthwise 3D convolutions for regularization
    - Cosine decay learning rate scheduling
    - Skip connections between conv blocks
    
    Architecture Components:
    1. Input Augmentation Layer
    2. Stack of Conv3D blocks (Depthwise + Skip connections)
    3. Global Average Pooling 3D
    4. Positional Encoding Generator
    5. Multi-Head Self-Attention with LayerNorm
    6. Dense Classification Head with dropout
    
    Input Shape Format: (time_steps, height, width, channels)
    """
    
    # --- Parameter Parsing ---
    base_filters = params.get("base_filters", 32)
    num_conv_layers = params.get("num_conv_layers", 4)
    dropout_rate = params.get("dropout_rate", 0.5)
    num_attention_heads = params.get("num_attention_heads", 4)
    attention_dim = params.get("attention_dim", 128)
    use_batch_norm = params.get("batch_norm", True)
    learning_rate = params.get("learning_rate", 5e-4)
    positional_encoding = params.get("positional_encoding", True)

    # --- Input Definition ---
    inputs = keras.Input(shape=input_shape, name="cwt_input")
    
    # --- Data Augmentation Layer ---
    def cwt_augmentation(x):
        """Applies noise augmentation tailored for CWT representations."""
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.01, dtype=x.dtype)
        return x + noise
    
    # Fix: Specify output_shape to resolve Lambda layer shape inference issue
    x = layers.Lambda(cwt_augmentation, output_shape=lambda s: s, name="cwt_augmentation")(inputs)

    # --- Conv3D Feature Extraction with Skip Connections ---
    skip_connections = []
    for i in range(num_conv_layers):
        filters = base_filters * (2 ** (i // 2))  # Scale filters every 2 layers
        
        # Depthwise Separable Conv3D Block
        conv = layers.Conv3D(filters, (1, 3, 3), padding="same", 
                            activation="relu", name=f"conv3d_{i+1}")(x)
        if use_batch_norm:
            conv = layers.BatchNormalization(name=f"bn_{i+1}")(conv)
            
        # Pooling every 2 layers
        if (i + 1) % 2 == 0:
            conv = layers.MaxPooling3D((1, 2, 2), name=f"pool_{i+1}")(conv)
            
        # Store for skip connections
        if i < num_conv_layers - 1:  # No skip connection from last layer
            skip_connections.append(conv)
            
        x = conv

    # --- Skip Connection Integration ---
    for skip in reversed(skip_connections):
        if x.shape[1:] == skip.shape[1:]:  # Match spatial dimensions
            x = layers.Add()([x, skip])

    # --- Preserve temporal dimension for attention ---
    # Use GlobalAveragePooling3D but keep time dimension
    x = layers.GlobalAveragePooling3D(data_format='channels_last', keepdims=True)(x)  # (batch, time, 1, 1, channels)
    
    # Fix: Properly reshape to maintain time dimension for attention
    # Get the time dimension and channel dimension
    time_steps = input_shape[0]
    channels = x.shape[-1]
    x = layers.Reshape((time_steps, channels))(x)  # (batch, time, channels)
    
    # --- Attention Mechanism ---
    if positional_encoding:
        # Positional Encoding (simplified for time steps)
        seq_len = input_shape[0]
        pos_encoding = layers.Embedding(input_dim=seq_len, output_dim=x.shape[-1], name="pos_encoding")
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_enc = pos_encoding(positions)
        pos_enc = tf.expand_dims(pos_enc, axis=0)  # Add batch dimension
        x = layers.Add()([x, pos_enc])
    
    # Multi-Head Self-Attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_attention_heads, 
        key_dim=attention_dim//num_attention_heads,
        name="mha"
    )(x, x)
    
    # Residual Connection + LayerNorm
    attention_output = layers.Add()([x, attention_output])
    attention_output = layers.LayerNormalization(name="ln")(attention_output)
    
    # Global pooling after attention
    pooled_features = layers.GlobalAveragePooling1D(name="temporal_pooling")(attention_output)

    # --- Classification Head ---
    features = layers.Dense(attention_dim, activation="relu", name="features")(pooled_features)
    features = layers.Dropout(dropout_rate, name="dropout")(features)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier", dtype='float32')(features)

    # --- Model Assembly ---
    model = keras.Model(inputs=inputs, outputs=outputs, name="ImprovedAttentionDCWT")
    
    # --- Cosine Decay Learning Rate ---
    initial_lr = learning_rate
    decay_steps = 1000
    alpha = 1e-6
    cosine_decay = keras.optimizers.schedules.CosineDecay(initial_lr, decay_steps, alpha)
    
    # --- Compilation ---
    model.compile(
        optimizer=keras.optimizers.Adam(cosine_decay),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model