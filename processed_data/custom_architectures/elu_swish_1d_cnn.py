import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape, num_classes, num_conv_layers=4, filters=32, kernel_size=7, dense_units=128, dropout_rate=0.5, learning_rate=0.001, weight_decay=1e-4):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Custom activation: ELU-Swish hybrid (ELU for negative, Swish for positive)
    def elu_swish(x):
        return tf.where(x < 0, tf.nn.elu(x), x * tf.nn.sigmoid(x))
    
    for i in range(num_conv_layers):
        x = layers.Conv1D(filters * (2 ** min(i, 2)), kernel_size, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(elu_swish)(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense_units, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(elu_swish)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model