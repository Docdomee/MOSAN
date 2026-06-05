import tensorflow as tf

def dual_branch_1d_cwt(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # First branch
    x1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x1 = tf.keras.layers.MaxPooling1D(pool_size=2)(x1)
    x1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x1)
    x1 = tf.keras.layers.MaxPooling1D(pool_size=2)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    
    # Second branch
    x2 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu')(inputs)
    x2 = tf.keras.layers.MaxPooling1D(pool_size=2)(x2)
    x2 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu')(x2)
    x2 = tf.keras.layers.MaxPooling1D(pool_size=2)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    
    # Merge branches
    merged = tf.keras.layers.concatenate([x1, x2])
    merged = tf.keras.layers.Dense(128, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(0.5)(merged)
    
    # Output
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(merged)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model
