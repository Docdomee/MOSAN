def build_model(input_shape, num_classes, params: dict):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    filters = params.get("filters", 100)
    kernel_size = params.get("kernel_size", 3)
    dropout_rate = params.get("dropout_rate", 0.0)
    learning_rate = params.get("learning_rate", 0.001)

    # Paper: each residual block has 4 conv layers with a shortcut connection
    # spanning the entire block. All strides are 1 — no downsampling anywhere.
    # "we do not use pooling layers and instead use strided convolutions with
    #  the goal of preserving the exact locations of spectral peaks" — but in
    # practice the paper uses stride=1 throughout (no spatial reduction at all).
    def residual_block(x, f, k):
        shortcut = x

        x = layers.Conv1D(f, k, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv1D(f, k, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv1D(f, k, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv1D(f, k, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)

        # Match channels on shortcut if first block changes filter count
        if shortcut.shape[-1] != f:
            shortcut = layers.Conv1D(f, 1, strides=1, padding='same', use_bias=False, kernel_initializer='he_normal')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

        return x

    inputs = keras.Input(shape=input_shape)

    # Layer 1: initial convolution — paper uses 64 filters here, then 100 in hidden layers
    x = layers.Conv1D(64, kernel_size=kernel_size, strides=1, padding='same',
                      use_bias=False, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 6 residual blocks x 4 conv layers = 24 conv layers + 1 initial = 25 total (paper Fig. 1c)
    for _ in range(6):
        x = residual_block(x, filters, kernel_size)

    # Paper: no intermediate pooling, GlobalAveragePooling only at the end
    x = layers.GlobalAveragePooling1D()(x)

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='resnet1d_nature_paper_faithful')

    # Paper: Adam with betas (0.5, 0.999) — this is the optimizer the authors
    # found necessary for stable training on noisy 1D spectral data.
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.5,
        beta_2=0.999
    )
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
