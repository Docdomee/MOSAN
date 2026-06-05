def build_model(input_shape, num_classes, params: dict):
    from tensorflow.keras import layers, models, regularizers
    import tensorflow as tf
    import numpy as np

    # Custom Attention Layer (Squeeze-Excitation for simplicity, combined with self-attention elements)
    class AttentionBlock(layers.Layer):
        def __init__(self, filters, name='attention_block'):
            super(AttentionBlock, self).__init__(name=name)
            self.filters = filters
            self.global_avg = layers.GlobalAveragePooling2D()
            self.dense1 = layers.Dense(filters // 16, activation='relu')
            self.dense2 = layers.Dense(filters, activation='sigmoid')
            self.multiply = layers.Multiply()

        def call(self, inputs):
            se = self.global_avg(inputs)
            se = self.dense1(se)
            se = self.dense2(se)
            se = tf.expand_dims(tf.expand_dims(se, axis=1), axis=1)
            return self.multiply([inputs, se])

    # Residual Block for 2D CNN
    def residual_block(x, filters, kernel_size=3, name_prefix=''):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    # Params defaults
    dropout_rate = params.get('dropout_rate', 0.3)
    num_residual_blocks = params.get('num_residual_blocks', 2)
    filters = params.get('filters', 64)

    # Model building (assuming input_shape is for 2D GAF, e.g., (height, width, 1))
    inputs = layers.Input(shape=input_shape)
    
    # Initial 2D Conv (no GAF transform needed since input is pre-computed 2D_GAF)
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Residual blocks
    for i in range(num_residual_blocks):
        x = residual_block(x, filters, name_prefix=f'residual_{i}')
    
    # Attention
    x = AttentionBlock(filters)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile with standard loss (removed entropy regularization to avoid serialization issues in multi-worker setups)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model