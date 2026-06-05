def build_model(input_shape=None, num_classes=None, params=None):
    """
    Builds the Keras model based on the provided parameters.
    Ensures compatibility with validation_worker.py by accepting 
    input_shape, num_classes, and params as keyword arguments.
    """
    from tensorflow.keras import layers, models

    # Handle case where params might be None
    if params is None:
        params = {}

    # Prioritize explicit arguments over values inside the params dictionary
    final_input_shape = input_shape if input_shape is not None else params.get("input_shape", (100, 1))
    final_num_classes = num_classes if num_classes is not None else params.get("num_classes", 1)
    
    num_filters = params.get("num_filters", 64)
    
    inputs = layers.Input(shape=final_input_shape)
    x = inputs

    # Example layer to establish initial dimensions
    x = layers.Conv1D(num_filters, 3, padding="same", activation="relu")(x)

    # Residual connection logic
    if params.get("use_residual", False) and not params.get("probe_phase_linear", True):
        shortcut = layers.Conv1D(num_filters, 1, padding="same")(x)
        x = layers.Add()([x, shortcut])

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(final_num_classes, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model