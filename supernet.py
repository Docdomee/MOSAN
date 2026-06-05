import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, backend as K
import numpy as np

# Define standard primitives
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# --- Dimensionality Helpers ---

def get_layer_factory(dim):
    """Returns a factory object/dict with layer constructors for the given dimension."""
    if dim == 1:
        return {
            'Conv': layers.Conv1D,
            'AvgPool': layers.AveragePooling1D,
            'MaxPool': layers.MaxPooling1D,
            'SepConv': layers.SeparableConv1D,
            'GlobalAvgPool': layers.GlobalAveragePooling1D,
            'ZeroPadding': layers.ZeroPadding1D, # Not always needed but good to have
        }
    elif dim == 2:
        return {
            'Conv': layers.Conv2D,
            'AvgPool': layers.AveragePooling2D,
            'MaxPool': layers.MaxPooling2D,
            'SepConv': layers.SeparableConv2D,
            'GlobalAvgPool': layers.GlobalAveragePooling2D,
        }
    elif dim == 3:
        return {
            'Conv': layers.Conv3D,
            'AvgPool': layers.AveragePooling3D,
            'MaxPool': layers.MaxPooling3D,
            'SepConv': layers.Conv3D, # FALLBACK: No SepConv3D in Keras default
            'GlobalAvgPool': layers.GlobalAveragePooling3D,
        }
    else:
        raise ValueError(f"Unsupported dimensionality: {dim}D (Only 1, 2, 3 supported)")

# --- Primitives Definitions ---
# Moved BEFORE get_ops_dict to ensure they are defined when lambdas capture them (best practice)

class Zero(layers.Layer):
    def __init__(self, stride, dim):
        super(Zero, self).__init__()
        self.stride = stride
        self.dim = dim

    def call(self, x):
        if self.stride == 1:
            return x * 0.
        
        # Slicing logic for N-D: x[:, ::stride, ::stride, ..., :]
        # Slice(None) for batch
        # Slice(None, None, stride) for each spatial dim
        # Slice(None) for channels
        slices = [slice(None)] + [slice(None, None, self.stride)] * self.dim + [slice(None)]
        return x[tuple(slices)] * 0.

class Identity(layers.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x):
        return x

class FactorizedReduce(layers.Layer):
    def __init__(self, C_out, dim, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = layers.ReLU()
        self.dim = dim
        L = get_layer_factory(dim)
        
        # Usa kernel 1x1, stride 2
        self.conv_1 = L['Conv'](C_out // 2, 1, strides=2, padding='valid', use_bias=False)
        self.conv_2 = L['Conv'](C_out // 2, 1, strides=2, padding='valid', use_bias=False)
        self.bn = layers.BatchNormalization()

    def call(self, x):
        x = self.relu(x)
        
        # --- FIX: Padding per gestire dimensioni dispari ---
        # Ensure dimensions are even before strictly strided operations
        shape = tf.shape(x)
        pad_params = [[0, 0]] # Batch
        for i in range(self.dim):
             d = shape[1 + i]
             # If d % 2 != 0 -> pad 1, else 0. This ensures we start with EVEN dims.
             p = d % 2
             pad_params.append([0, p])
        pad_params.append([0, 0]) # Channel
        
        x = tf.pad(x, pad_params)
        # -------------------------------------------------

        # Shifted input: x[:, 1:, 1:, ..., :] 
        slices = [slice(None)] + [slice(1, None)] * self.dim + [slice(None)]
        x_shifted = x[tuple(slices)]
        
        # Convolutions
        out1 = self.conv_1(x)
        out2 = self.conv_2(x_shifted)
        
        # --- FIX: Robust Output Sizing ---
        # Ensure spatial dimensions match exactly before concatenation.
        # This handles edge cases where padding/shifting logic might still cause off-by-one by subtle arithmetic.
        s1 = tf.shape(out1)
        s2 = tf.shape(out2)
        
        slice_params = [slice(None)] # Batch
        for i in range(self.dim):
            min_l = tf.minimum(s1[1+i], s2[1+i])
            slice_params.append(slice(0, min_l))
        slice_params.append(slice(None)) # Channel
        
        out1 = out1[tuple(slice_params)]
        out2 = out2[tuple(slice_params)]
        # ---------------------------------

        out = tf.concat([out1, out2], axis=-1)
        return self.bn(out)

class ReLUConvBN(layers.Layer):
    def __init__(self, C_out, kernel_size, stride, padding, dim, affine=True):
        super(ReLUConvBN, self).__init__()
        L = get_layer_factory(dim)
        
        self.op = keras.Sequential([
            layers.ReLU(),
            L['Conv'](C_out, kernel_size, strides=stride, padding='same', use_bias=False),
            layers.BatchNormalization()
        ])

    def call(self, x):
        return self.op(x)

class DilConv(layers.Layer):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, dim, affine=True):
        super(DilConv, self).__init__()
        L = get_layer_factory(dim)
        
        # Fix for Keras: strides > 1 not supported with dilation_rate > 1 in some layers?
        # For Conv2D it is supported in recent TF? No, usually not.
        depthwise_stride = 1 if dilation > 1 else stride
        pointwise_stride = stride if dilation > 1 else 1
        
        seq = [layers.ReLU()]
        
        # Depthwise / Separable
        if dim == 3 and L['SepConv'] == layers.Conv3D:
            # Fallback for 3D: Use standard Conv3D with dilation
            seq.append(L['Conv'](C_in, kernel_size, strides=depthwise_stride, padding='same', dilation_rate=dilation, use_bias=False)) 
        else:
            # 1D / 2D use SeparableConv (which performs depthwise then pointwise usually, or just depthwise?)
            # Keras SeparableConvND is actually Depthwise + Pointwise.
            # But DilConv usually requires *Depthwise* convolution with dilation.
            # Keras SeparableConvND does NOT support dilation_rate != 1 easily in old versions? 
            # Check docs: SeparableConv2D supports dilation_rate.
            seq.append(L['SepConv'](C_in, kernel_size, strides=depthwise_stride, padding='same', dilation_rate=dilation, use_bias=False))
        
        seq.append(layers.BatchNormalization())
        
        # Pointwise projection
        seq.append(L['Conv'](C_out, 1, strides=pointwise_stride, padding='same', use_bias=False))
        seq.append(layers.BatchNormalization())
        
        self.op = keras.Sequential(seq)

    def call(self, x):
        return self.op(x)

class SepConv(layers.Layer):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dim, affine=True):
        super(SepConv, self).__init__()
        L = get_layer_factory(dim)
        
        seq = [layers.ReLU()]
        
        # First SepConv
        if dim == 3 and L['SepConv'] == layers.Conv3D:
             seq.append(L['Conv'](C_in, kernel_size, strides=stride, padding='same', use_bias=False))
        else:
             seq.append(L['SepConv'](C_in, kernel_size, strides=stride, padding='same', use_bias=False))
             
        seq.append(layers.BatchNormalization())
        
        # Second SepConv (Stacking them as per NASNet/DARTS design)
        # stride=1 for the second one usually?
        if dim == 3 and L['SepConv'] == layers.Conv3D:
             seq.append(L['Conv'](C_in, kernel_size, strides=1, padding='same', use_bias=False))
        else:
             seq.append(L['SepConv'](C_in, kernel_size, strides=1, padding='same', use_bias=False))
             
        seq.append(layers.BatchNormalization())
        
        self.op = keras.Sequential(seq)

    def call(self, x):
        return self.op(x)

def get_ops_dict(dim):
    """Returns the OPS dictionary specialized for the given dimension."""
    L = get_layer_factory(dim)
    
    # helper for pooling padding='same'
    def pool(type_name, k, stride):
        # 3x3 for 2D, 3 for 1D, 3x3x3 for 3D logic
        # Ideally passed k is just an integer '3' -> applies to all dims
        return L[type_name](k, strides=stride, padding='same')

    # Explicit definitions for clear traceback
    def op_none(C, stride, affine): return Zero(stride, dim)
    def op_avg_pool(C, stride, affine): return pool('AvgPool', 3, stride)
    def op_max_pool(C, stride, affine): return pool('MaxPool', 3, stride)
    def op_skip_connect(C, stride, affine): 
        # print(f"[DEBUG] op_skip_connect: C={C}, stride={stride}, dim={dim}")
        if stride == 1:
            return Identity()
        else:
            return FactorizedReduce(C, dim, affine=affine)
    def op_sep_3x3(C, stride, affine): return SepConv(C, C, 3, stride, 1, dim, affine=affine)
    def op_sep_5x5(C, stride, affine): return SepConv(C, C, 5, stride, 2, dim, affine=affine)
    def op_dil_3x3(C, stride, affine): return DilConv(C, C, 3, stride, 2, 2, dim, affine=affine)
    def op_dil_5x5(C, stride, affine): return DilConv(C, C, 5, stride, 4, 2, dim, affine=affine)

    return {
        'none': op_none,
        'avg_pool_3x3': op_avg_pool,
        'max_pool_3x3': op_max_pool,
        'skip_connect': op_skip_connect,
        'sep_conv_3x3': op_sep_3x3,
        'sep_conv_5x5': op_sep_5x5,
        'dil_conv_3x3': op_dil_3x3,
        'dil_conv_5x5': op_dil_5x5,
    }

class MixedOperation(layers.Layer):
    def __init__(self, C, stride, dim, operation_space=None):
        super(MixedOperation, self).__init__()
        self._ops = []
        self.operation_space = operation_space if operation_space else PRIMITIVES
        
        # Get ops specialized for this dimension
        ops_dict = get_ops_dict(dim)
        
        for primitive in self.operation_space:
            op = ops_dict[primitive](C, stride, False)
            self._ops.append(op)

    def call(self, x, weights):
        # Cast weights to match tensor dtype (fixes float16/float32 mismatch in mixed precision)
        weights = tf.cast(weights, x.dtype)
        return tf.add_n([weights[i] * self._ops[i](x) for i in range(len(self._ops))])

class Cell(layers.Layer):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, dim, operation_space=None):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.operation_space = operation_space
        self.dim = dim

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C, dim=dim, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C, 1, 1, 0, dim=dim, affine=False)
            
        self.preprocess1 = ReLUConvBN(C, 1, 1, 0, dim=dim, affine=False)
        
        self._steps = steps
        self._multiplier = multiplier
        self._ops = []
        
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOperation(C, stride, dim=dim, operation_space=self.operation_space)
                self._ops.append(op)

    def call(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
            
        return tf.concat(states[-self._multiplier:], axis=-1)

class Supernet(keras.Model):
    def __init__(self, input_shape, num_classes, layers_count=8, steps=4, multiplier=4, stem_multiplier=3, init_channels=16, operation_space=None, hypothesis_context=None):
        super(Supernet, self).__init__()
        
        # Infer dimensionality from input_shape
        # input_shape typically is (H, W, C) for 2D, (L, C) for 1D, (D, H, W, C) for 3D.
        # We assume channel is last.
        # dim = total ranks - 1 (for channel).
        self.spatial_dim = len(input_shape) - 1
        print(f"DEBUG: Initializing Supernet for {self.spatial_dim}D data. Shape: {input_shape}")
        
        self._C = init_channels # Initial channels
        self._num_classes = num_classes
        self.layers_count = layers_count
        self._steps = steps
        self._multiplier = multiplier
        self.operation_space = operation_space if operation_space else PRIMITIVES
        self.hypothesis_context = hypothesis_context

        C_curr = self._steps * self._multiplier * self._C
        
        L = get_layer_factory(self.spatial_dim)
        
        self.stem = keras.Sequential([
            L['Conv'](C_curr, 3, padding='same', use_bias=False),
            layers.BatchNormalization()
        ])
        
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self._C
        self.cells = []
        reduction_prev = False
        
        for i in range(self.layers_count):
            if i in [self.layers_count // 3, 2 * self.layers_count // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
                
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, dim=self.spatial_dim, operation_space=self.operation_space)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = L['GlobalAvgPool']()
        self.classifier_dense = layers.Dense(num_classes)
        self.classifier_activation = layers.Activation('softmax', dtype='float32')

        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(self.operation_space)
        
        # FIX: Distributed Strategy Compatibility
        # We must explicitly set aggregation to MEAN for these trainable variables 
        # so MirroredStrategy knows how to sync their gradients across replicas.
        # Without this, or if created outside scope, we get "access resource from diff device" errors.
        
        self.alphas_normal = tf.Variable(
            initial_value=1e-3 * tf.random.normal((k, num_ops)), 
            name='alphas_normal', 
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN # Critical for Multi-GPU
        )
        
        self.alphas_reduce = tf.Variable(
            initial_value=1e-3 * tf.random.normal((k, num_ops)), 
            name='alphas_reduce', 
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN # Critical for Multi-GPU
        )
        
        self._arch_parameters = [self.alphas_normal, self.alphas_reduce]

    def call(self, inputs):
        s0 = s1 = self.stem(inputs)
        
        weights_normal = tf.nn.softmax(self.alphas_normal, axis=-1)
        weights_reduce = tf.nn.softmax(self.alphas_reduce, axis=-1)
        
        for cell in self.cells:
            if cell.reduction:
                weights = weights_reduce
            else:
                weights = weights_normal
            s0, s1 = s1, cell(s0, s1, weights)
            
        out = self.global_pooling(s1)
        out = self.classifier_dense(out)
        logits = self.classifier_activation(out)
        return logits

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if self.operation_space[k] != 'none'))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if self.operation_space[k] != 'none':
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((self.operation_space[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(tf.nn.softmax(self.alphas_normal, axis=-1).numpy())
        gene_reduce = _parse(tf.nn.softmax(self.alphas_reduce, axis=-1).numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        return {'normal': gene_normal, 'normal_concat': list(concat), 'reduce': gene_reduce, 'reduce_concat': list(concat)}

    def entropy(self):
        """Calculates the mean entropy of the architecture parameters (alphas)."""
        probs_normal = tf.nn.softmax(self.alphas_normal, axis=-1)
        probs_reduce = tf.nn.softmax(self.alphas_reduce, axis=-1)
        
        entropy_normal = -tf.reduce_sum(probs_normal * tf.math.log(probs_normal + 1e-10), axis=-1)
        entropy_reduce = -tf.reduce_sum(probs_reduce * tf.math.log(probs_reduce + 1e-10), axis=-1)
        
        return tf.reduce_mean(entropy_normal) + tf.reduce_mean(entropy_reduce)

def build_supernet(input_shape, num_classes, params=None):
    if params is None:
        params = {}
    
    return Supernet(
        input_shape=input_shape,
        num_classes=num_classes,
        layers_count=params.get('layers', 8),
        steps=params.get('steps', 4),
        multiplier=params.get('multiplier', 4),
        init_channels=params.get('init_channels', 16),
        operation_space=params.get('operation_space', None),
        hypothesis_context=params.get('hypothesis_context', None)
    )
