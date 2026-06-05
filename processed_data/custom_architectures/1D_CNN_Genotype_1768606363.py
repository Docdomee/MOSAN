
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import supernet 
import json

class DropPath(layers.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def call(self, x, training=None):
        if training and self.drop_prob > 0.:
            keep_prob = 1. - self.drop_prob
            # FIX: Usiamo len(x.shape) che è statico, non tf.shape(x) che è simbolico
            shape = (tf.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, dtype=x.dtype)
            binary_tensor = tf.floor(random_tensor)
            x = tf.divide(x, keep_prob) * binary_tensor
        return x

class DiscreteCell(layers.Layer):
    def __init__(self, c_prev_prev, c_prev, c_curr, reduction, reduction_prev, dim, genotype, reg=None):
        super(DiscreteCell, self).__init__()
        self.reduction = reduction
        self.dim = dim
        self._concat = genotype['reduce_concat'] if reduction else genotype['normal_concat']
        self._multiplier = len(self._concat) 
        
        if reduction_prev:
            self.preprocess0 = supernet.FactorizedReduce(c_curr, dim=dim, affine=False)
        else:
            self.preprocess0 = supernet.ReLUConvBN(c_curr, 1, 1, 0, dim=dim, affine=False)
        self.preprocess1 = supernet.ReLUConvBN(c_curr, 1, 1, 0, dim=dim, affine=False)
        
        op_names = genotype['reduce'] if reduction else genotype['normal']
        self._ops = []
        self._indices = []
        
        ops_dict = supernet.get_ops_dict(dim)
        for name, index in op_names:
            stride = 2 if reduction and index < 2 else 1
            op = ops_dict[name](c_curr, stride, True)
            self._ops.append(op)
            self._indices.append(index)
        
        self.drop_path = DropPath(0.1)

    def call(self, s0, s1, training=None):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        
        for i in range(len(self._ops) // 2):
            h1 = states[self._indices[2*i]]
            op1 = self._ops[2*i]
            h2 = states[self._indices[2*i+1]]
            op2 = self._ops[2*i+1]
            
            out1 = op1(h1)
            out2 = op2(h2)
            if training:
                out1 = self.drop_path(out1, training=training)
                out2 = self.drop_path(out2, training=training)
            
            s = out1 + out2
            states.append(s)
        return tf.concat([states[i] for i in self._concat], axis=-1)

def build_model(input_shape, num_classes, params):
    dim = len(input_shape) - 1
    c_curr = params.get('init_channels', 36)
    layers_count = params.get('layers', 8)
    genotype = {'normal': [['skip_connect', 0], ['skip_connect', 1], ['skip_connect', 2], ['sep_conv_5x5', 0], ['max_pool_3x3', 0], ['max_pool_3x3', 2], ['skip_connect', 0], ['skip_connect', 2]], 'normal_concat': [2, 3, 4, 5], 'reduce': [['max_pool_3x3', 1], ['avg_pool_3x3', 0], ['dil_conv_5x5', 2], ['max_pool_3x3', 0], ['dil_conv_5x5', 3], ['dil_conv_5x5', 0], ['skip_connect', 4], ['max_pool_3x3', 1]], 'reduce_concat': [2, 3, 4, 5]}
    
    l2 = params.get("l2_regularization", 1e-4)
    reg = keras.regularizers.l2(l2) if l2 > 0 else None

    L = supernet.get_layer_factory(dim)
    inputs = keras.Input(shape=input_shape)
    
    x = L['Conv'](c_curr * 3, 3, padding='same', use_bias=False, kernel_regularizer=reg)(inputs)
    x = layers.BatchNormalization()(x)
    s0 = s1 = x

    reduction_prev = False
    for i in range(layers_count):
        reduction = i in [layers_count // 3, 2 * layers_count // 3]
        cell = DiscreteCell(None, None, c_curr, reduction, reduction_prev, dim, genotype, reg=reg)
        s0, s1 = s1, cell(s0, s1)
        reduction_prev = reduction
        if reduction: c_curr *= 2

    out = L['GlobalAvgPool']()(s1)
    if params.get("dropout_rate", 0) > 0:
        out = layers.Dropout(params.get("dropout_rate"))(out)
        
    outputs = layers.Dense(num_classes, activation="softmax", kernel_regularizer=reg)(out)
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params.get("learning_rate", 1e-3)),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
