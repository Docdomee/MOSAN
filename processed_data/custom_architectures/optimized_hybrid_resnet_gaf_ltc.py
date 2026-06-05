def build_model(input_shape, num_classes, params):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import numpy as np
    
    # Define the LTC cell properly
    class LTCell(keras.layers.AbstractRNNCell):
        def __init__(self, units, **kwargs):
            super(LTCell, self).__init__(**kwargs)
            self.units = units
            self.state_size = units  # Fixed: placed inside __init__ method
            
        def build(self, input_shape):
            # Initialize ODE solver weights
            self.w_input = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                name='w_input'
            )
            self.w_hidden = self.add_weight(
                shape=(self.units, self.units),
                initializer='glorot_uniform',
                name='w_hidden'
            )
            self.built = True
            
        def call(self, inputs, states):
            prev_output = states[0]
            # Simple linear transformation as placeholder ODE
            h = tf.matmul(inputs, self.w_input) + tf.matmul(prev_output, self.w_hidden)
            # Apply tanh activation
            output = tf.nn.tanh(h)
            return output, [output]
        
        def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
            return [tf.zeros((batch_size, self.units), dtype=dtype)]
            
        @property
        def output_size(self):
            return self.units

    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # ResNet blocks for feature extraction from 2D GAF
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Additional ResNet-inspired blocks
    for i in range(2):
        residual = x
        x = layers.Conv2D(64 * (i+1), (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64 * (i+1), (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust residual connection to match dimensions
        if i == 0:
            # For first block, residual has 32 channels, x has 64 channels
            residual = layers.Conv2D(64, (1, 1), padding='same')(residual)
        elif i == 1:
            # For second block, residual already has 64 channels, x has 128 channels
            residual = layers.Conv2D(128, (1, 1), padding='same')(residual)
            
        x = layers.Add()([x, residual])
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
    
    # Prepare for RNN: reshape to (batch_size, timesteps, features)
    shape = tf.shape(x)
    x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)
    
    # LTC RNN layer - using SimpleRNN with custom cell
    ltc_cell = LTCell(units=64)
    x = layers.RNN(ltc_cell, return_sequences=False)(x)
    
    # Final classification layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model