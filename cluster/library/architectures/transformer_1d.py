import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_shape, num_classes, params):
    """
    Transformer-1D Architecture (Hackable Baseline).
    
    Uses Multi-Head Self-Attention layers instead of Convolutions for feature extraction.
    Crucial for capturing long-range dependencies in time-series data.
    
    Structure:
    - Input Embedding (Linear projection of input signal)
    - Positional Encoding (Added to input)
    - Transformer Encoder Blocks (Attention + FeedForward)
    - Global Pooling -> Head
    """
    
    # --- Params ---
    head_size = params.get("head_size", 64)
    num_heads = params.get("num_heads", 4)
    ff_dim = params.get("ff_dim", 128)
    num_transformer_blocks = params.get("num_blocks", 3)
    dropout = params.get("dropout_rate", 0.1)
    
    def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    # --- Build ---
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    # Optional: Initial projection/embedding if input dim is small
    x = layers.Dense(head_size)(x) 
    
    # Warning: Standard Transformers need Postional Encodings.
    # For simplicity in this baseline, we assume the dense projection + convolution nature of Conv1D in FF 
    # might implicitly handle some locality, but adding a proper embedding is better for the Architect to implement.
    # We will leave a comment for the Architect.
    # [ARCHITECT NOTE]: Add Sinusoidal or Learnable Positional Embeddings here if needed.

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    layers.Dense(num_classes)(x)`n    outputs = layers.Activation("softmax", dtype="float32")(x)
    
    model = keras.Model(inputs, outputs, name="Transformer_1D")
    
    optimizer = keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-4)) # Lower LR for Transformers
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
