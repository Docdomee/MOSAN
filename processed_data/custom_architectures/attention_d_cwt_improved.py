def build_model(input_shape, num_classes, params):
    """
    Improved attention_3d_cwt model with advanced regularization, architectural refinements,
    and distribution shift robustness enhancements.

    Hypothesis:
    - Retain the core 3D Conv + GAP + Attention structure from the baseline.
    - Enhance regularization: moderate dropout (0.4-0.5) with spatial dropout,
      advanced L2 regularization tuning, and optional Stochastic Depth.
    - Refine attention block: explicit residual connections, layer norm, and proper scaling.
    - Modest capacity increase with filter scaling and LR adjustment.
    - Stronger generalization focus through architectural simplicity in the head.
    
    Parameters:
    - filters (int): Base filter count; controls model capacity (default: 48).
    - dropout_rate (float): Dropout rate applied to conv and dense layers (default: 0.45).
    - attention_dim (int): Total attention dimension (default: 128).
    - num_heads (int): Number of attention heads (default: 4).
    - lr (float): Learning rate (default: 0.0004).
    - weight_decay (float): L2 regularization for kernels (default: 3e-5).
    - use_stochastic_depth (bool): Enables stochastic depth for residual blocks (default: False).
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    # --- Custom Stochastic Depth Implementation ---
    class StochasticDepth(layers.Layer):
        def __init__(self, survival_probability=0.8, **kwargs):
            super().__init__(**kwargs)
            self.survival_probability = survival_probability
            
        def call(self, inputs, training=None):
            if not training:
                return inputs[0] + inputs[1] * self.survival_probability
                
            # Generate random uniform tensor with same shape as inputs
            batch_size = tf.shape(inputs[0])[0]
            random_tensor = tf.random.uniform([batch_size, 1, 1], dtype=tf.float32)
            binary_tensor = tf.cast(random_tensor < self.survival_probability, tf.float32)
            
            # Scale during training to maintain expected value
            scaled_binary_tensor = binary_tensor / self.survival_probability
            
            return inputs[0] + inputs[1] * scaled_binary_tensor
            
        def get_config(self):
            config = super().get_config()
            config.update({"survival_probability": self.survival_probability})
            return config

    # --- Hyperparameters ---
    filters = params.get("filters", 48)
    dropout_rate = params.get("dropout_rate", 0.45)
    attention_dim = params.get("attention_dim", 128)
    num_heads = params.get("num_heads", 4)
    learning_rate = params.get("lr", 0.0004)
    weight_decay = params.get("weight_decay", 3e-5)
    use_stochastic_depth = params.get("use_stochastic_depth", False)

    key_dim = attention_dim // num_heads

    # --- Model Definition ---
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # --- 3D Convolutional Backbone ---
    for i in range(3):
        x = layers.Conv3D(
            filters=filters * (2 ** i),
            kernel_size=(3, 3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(weight_decay),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SpatialDropout3D(dropout_rate * 0.5)(x)  # Scaled spatial dropout
        if i < 2:
            x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # --- Global Average Pooling to reduce spatial dims ---
    x = layers.GlobalAveragePooling3D()(x)  # Now shape: (batch, features)

    # For 3D_DYNAMIC_GAF, we need to treat the feature dimension as a sequence
    # Reshape to add a sequence dimension
    feature_dim = x.shape[-1]
    
    # Ensure attention_dim matches feature_dim for proper reshaping
    if feature_dim != attention_dim:
        # Project to attention dimension if needed
        x = layers.Dense(attention_dim)(x)
        feature_dim = attention_dim
    
    # Reshape to treat features as a sequence
    # We'll treat the features as multiple "time steps" of smaller dimension
    seq_len = attention_dim // 16  # e.g., if attention_dim=128, seq_len=8
    feature_per_step = 16  # e.g., 128//8 = 16 features per time step
    
    x = layers.Reshape((seq_len, feature_per_step))(x)

    # --- Attention Block with Residual and LayerNorm ---
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate * 0.8,
    )(x, x)

    # Optional Stochastic Depth for regularization
    if use_stochastic_depth:
        attention_output = StochasticDepth(survival_probability=0.8)([x, attention_output])
    else:
        attention_output = layers.Add()([x, attention_output])

    x = layers.LayerNormalization()(attention_output)

    # --- Feed Forward with Residual ---
    ff_dim = attention_dim * 2
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dropout(dropout_rate)(ff)
    ff = layers.Dense(attention_dim)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)

    # --- Pooling over time for classification ---
    x = layers.GlobalAveragePooling1D()(x)

    # --- Dense Head with strong L2 and dropout ---
    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay * 10),
    )(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="attention_3d_cwt_improved")

    # --- Compiler ---
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model