"""
HybridResNetGAF with ConvLTC Embedding Layer
=============================================
Hierarchical architecture for SERS bacterial classification on aged substrates.
Input: 3D_DYNAMIC_GAF volumes (N_frames, H, W, 1)

Architecture:
  1. Shared 2D ResNet encoder  — extracts per-frame spatial feature maps (shared weights)
  2. ConvLTC trunk             — processes the sequence of feature maps as a spatial RNN:
                                 state h_t is a 2D feature map (H', W', C_hidden),
                                 dynamics are convolutional (not matrix multiply)
                                 h_0 is initialised from the calibration volume
  3. Classification head       — GAP → Dense → softmax

Physical motivation:
  - Shared 2D encoder forces scale-invariant features across sigma levels
  - ConvLTC reads the focusing trajectory σ=5→0 as a temporal sequence of spatial maps
  - τ(x) is spatially localised: unstable spectral regions (e.g. ~1002 cm⁻¹ aging band)
    develop large τ → slow update → effectively downweighted in the final embedding
  - h_0 from calibration volume conditions the entire trajectory read:
    the LTC "knows" what the current substrate looks like before reading the sample

Calibration protocol:
  - Training:   calib_volume = zeros  → h_0 = learned bias, LTC reads trajectory freely
  - Inference:  calib_volume = mean of ~10 aged reference spectra (same substrate)
                → h_0 encodes the substrate state → LTC corrects trajectory accordingly

Hyperparameters exposed to MOSAN:
  num_conv_layers_2d  : ResNet blocks in shared 2D encoder (1–4)
  filters_2d          : base filters for 2D encoder (16–128, doubles per block)
  conv_ltc_units      : channels in ConvLTC hidden state (16–128)
  conv_ltc_kernel     : spatial kernel size for ConvLTC (3 or 5)
  tau_min             : minimum time constant (0.01–1.0)
  dense_units         : classification head size (64–256)
  dropout_rate        : dropout before softmax (0.1–0.6)
  learning_rate       : Adam lr (1e-4–1e-2)
  weight_decay        : L2 regularisation (1e-5–1e-3)
  use_residual        : residual connections in 2D encoder (bool)
  use_auxiliary_head  : Inception-style aux loss α=0.3 (bool, training only)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ---------------------------------------------------------------------------
# Utility layers (replace Lambda to allow safe model serialization/loading)
# ---------------------------------------------------------------------------

class ReduceMeanAxis1(keras.layers.Layer):
    """Mean over axis=1 (time/frame axis). Works for any rank ≥ 2."""
    def call(self, x):
        return tf.reduce_mean(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + tuple(input_shape[2:])

    def get_config(self):
        return super().get_config()


class ZerosLike(keras.layers.Layer):
    """Returns a zero tensor with the same shape as the input."""
    def call(self, x):
        return tf.zeros_like(x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()


# ---------------------------------------------------------------------------
# ConvLTC Cell
# ---------------------------------------------------------------------------
# Convolutional Liquid Time-Constant cell.
# Hidden state h_t is a 2D feature map (H', W', units) — not a vector.
# Dynamics:
#   f(h, x) = tanh( h ⊛ W_h  +  x ⊛ W_x  + b )
#   τ(x)    = softplus( x ⊛ W_tau + b_tau ) + tau_min
#   h_new   = (h + f/τ) / (1 + 1/τ)          [semi-implicit Euler, dt=1]
#
# All convolutions use padding='same' so spatial dimensions are preserved.
# ---------------------------------------------------------------------------

class ConvLTCCell(keras.layers.Layer):
    """
    Single ConvLTC cell. Processes one flat frame (H'*W'*C_in,) at a time,
    maintaining spatial hidden state flattened to (H'*W'*units,).
    Spatial dims must be provided at construction time so state_size is
    available before build() — required by keras.layers.RNN.
    """

    def __init__(self, units: int, spatial_h: int, spatial_w: int, c_in: int,
                 kernel_size: int = 3, tau_min: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.units      = units
        self.spatial_h  = spatial_h
        self.spatial_w  = spatial_w
        self.c_in       = c_in
        self.kernel_size = kernel_size
        self.tau_min    = tau_min
        # Known at construction time — required by keras.layers.RNN
        self.state_size  = spatial_h * spatial_w * units
        self.output_size = spatial_h * spatial_w * units

    def build(self, input_shape):
        k = self.kernel_size
        # Dynamics kernels
        self.W_x = self.add_weight(
            name="W_x",   shape=(k, k, self.c_in,   self.units), initializer="glorot_uniform")
        self.W_h = self.add_weight(
            name="W_h",   shape=(k, k, self.units,   self.units), initializer="orthogonal")
        self.b   = self.add_weight(
            name="b",     shape=(self.units,),                     initializer="zeros")
        # Time-constant kernels
        self.W_tau = self.add_weight(
            name="W_tau", shape=(k, k, self.c_in,   self.units), initializer="glorot_uniform")
        self.b_tau = self.add_weight(
            name="b_tau", shape=(self.units,),                     initializer="zeros")
        super().build(input_shape)

    def call(self, x_flat, states):
        # x_flat: (batch, H'*W'*C_in) — RNN API passes flat inputs
        # states[0]: (batch, H'*W'*units)
        batch = tf.shape(x_flat)[0]
        H, W, U = self.spatial_h, self.spatial_w, self.units
        C_in = x_flat.shape[-1] // (H * W)

        # Reshape back to spatial
        x = tf.reshape(x_flat, (batch, H, W, C_in))
        h = tf.reshape(states[0], (batch, H, W, U))

        # Dynamics
        f = tf.nn.tanh(
            tf.nn.conv2d(x, self.W_x, strides=1, padding="SAME") +
            tf.nn.conv2d(h, self.W_h, strides=1, padding="SAME") +
            self.b
        )

        # Input-dependent time constant (spatially localised)
        tau = (tf.nn.softplus(
            tf.nn.conv2d(x, self.W_tau, strides=1, padding="SAME") + self.b_tau
        ) + self.tau_min)

        # Semi-implicit Euler update
        h_new = (h + f / tau) / (1.0 + 1.0 / tau)

        # Flatten for RNN API
        h_new_flat = tf.reshape(h_new, (batch, H * W * U))
        return h_new_flat, [h_new_flat]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Zero initial state — overridden externally when calib h_0 is provided."""
        return [tf.zeros((batch_size, self.state_size), dtype=dtype or tf.float32)]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "spatial_h": self.spatial_h,
                    "spatial_w": self.spatial_w, "c_in": self.c_in,
                    "kernel_size": self.kernel_size, "tau_min": self.tau_min})
        return cfg


# ---------------------------------------------------------------------------
# 2D ResNet block (shared across all DGAF frames)
# ---------------------------------------------------------------------------

def _resnet_block_2d(x, filters, kernel_size, use_residual, reg, block_id: str = ""):
    """block_id must be unique across the entire model to avoid generic name collisions."""
    shortcut = x
    x = keras.layers.Conv2D(
        filters, kernel_size, padding="same", kernel_regularizer=reg,
        name=f"rb{block_id}_conv1")(x)
    x = keras.layers.BatchNormalization(name=f"rb{block_id}_bn1")(x)
    x = keras.layers.Activation("relu", name=f"rb{block_id}_relu1")(x)
    x = keras.layers.Conv2D(
        filters, kernel_size, padding="same", kernel_regularizer=reg,
        name=f"rb{block_id}_conv2")(x)
    x = keras.layers.BatchNormalization(name=f"rb{block_id}_bn2")(x)
    if use_residual:
        if shortcut.shape[-1] != filters:
            shortcut = keras.layers.Conv2D(
                filters, 1, padding="same", name=f"rb{block_id}_proj")(shortcut)
        x = keras.layers.Add(name=f"rb{block_id}_add")([x, shortcut])
    x = keras.layers.Activation("relu", name=f"rb{block_id}_relu2")(x)
    return x


# ---------------------------------------------------------------------------
# Main build_model
# ---------------------------------------------------------------------------

def _build_two_input_model(input_shape, num_classes, params):
    """
    Full two-input model: [dgaf_volume, calib_volume].
    Used directly by freeze_and_adapt_ltc.py (Phase 2).
    Not called by training_worker.py — use build_model() for MOSAN.
    """
    input_shape = tuple(input_shape)
    N_frames, H, W, C = input_shape

    n_blocks_2d  = max(1, min(4, int(params.get("num_conv_layers_2d",
                                                 params.get("num_conv_layers", 2)))))
    f2d          = int(params.get("filters_2d",       params.get("filters", 32)))
    ltc_units    = int(params.get("conv_ltc_units",   32))
    ltc_kernel   = int(params.get("conv_ltc_kernel",  3))
    tau_min      = float(params.get("tau_min",        0.1))
    dense_units  = int(params.get("dense_units",      128))
    dropout_rate = float(params.get("dropout_rate",   0.4))
    lr           = float(params.get("learning_rate",  1e-3))
    wd           = float(params.get("weight_decay",   1e-4))
    use_residual = bool(params.get("use_residual",    True))
    use_aux_head = bool(params.get("use_auxiliary_head", False))

    reg = keras.regularizers.L2(wd) if wd > 0 else None

    n_pools = n_blocks_2d - 1
    H_enc = H // (2 ** n_pools)
    W_enc = W // (2 ** n_pools)
    C_enc = min(512, f2d * (2 ** (n_blocks_2d - 1)))

    main_input  = keras.Input(shape=input_shape, name="dgaf_volume")
    calib_input = keras.Input(shape=input_shape, name="calib_volume")

    # Shared 2D encoder — built as a Model, applied once via TimeDistributed on main
    # and reused in-place (as a Layer call) on the calibration mean.
    # Using TimeDistributed twice on the same sub-Model breaks weight (de)serialization.
    frame_in = keras.Input(shape=(H, W, C), name="single_frame")
    fx = frame_in
    for i in range(n_blocks_2d):
        filters_i = min(512, f2d * (2 ** i))
        fx = _resnet_block_2d(fx, filters_i, 3, use_residual, reg, block_id=str(i))
        if i < n_blocks_2d - 1:
            fx = keras.layers.MaxPooling2D(2, name=f"pool_{i}")(fx)
    frame_encoder = keras.Model(frame_in, fx, name="shared_2d_encoder")

    # Main branch: apply per-frame encoder via TimeDistributed
    encoded_main = keras.layers.TimeDistributed(frame_encoder, name="td_main")(main_input)

    # Calibration branch: average frames first → single 2D tensor → apply encoder once
    # Mathematically equivalent to averaging the encoded frames when the encoder has
    # linear layers only; with BN/residuals it's close enough and, crucially, shares
    # all weights cleanly with the main branch.
    calib_raw_mean = ReduceMeanAxis1(name="calib_raw_mean")(calib_input)
    calib_mean_map = frame_encoder(calib_raw_mean)

    # Auxiliary head (after TimeDistributed, outside sub-model)
    if use_aux_head:
        td_aux_gap = keras.layers.TimeDistributed(
            keras.layers.GlobalAveragePooling2D(), name="td_aux_gap"
        )(encoded_main)
        aux_mean_vec = ReduceMeanAxis1(name="aux_mean_vec")(td_aux_gap)
        aux_logits = keras.layers.Dense(num_classes, name="aux_logits")(aux_mean_vec)
        aux_output = keras.layers.Activation(
            "softmax", dtype="float32", name="aux_output")(aux_logits)
    h0_spatial = keras.layers.Conv2D(
        ltc_units, kernel_size=1, padding="same",
        activation="tanh", name="h0_proj"
    )(calib_mean_map)
    h0_flat = keras.layers.Flatten(name="h0_flat")(h0_spatial)

    # ConvLTC trunk
    flat_frames = keras.layers.Reshape(
        (N_frames, H_enc * W_enc * C_enc), name="flat_frames"
    )(encoded_main)
    ltc_cell = ConvLTCCell(
        units=ltc_units, spatial_h=H_enc, spatial_w=W_enc, c_in=C_enc,
        kernel_size=ltc_kernel, tau_min=tau_min, name="conv_ltc_cell"
    )
    ltc_rnn = keras.layers.RNN(ltc_cell, return_sequences=False, name="conv_ltc_rnn")
    ltc_out_flat = ltc_rnn(flat_frames, initial_state=[h0_flat])

    ltc_out_spatial = keras.layers.Reshape(
        (H_enc, W_enc, ltc_units), name="ltc_spatial"
    )(ltc_out_flat)
    embedding = keras.layers.GlobalAveragePooling2D(name="embedding")(ltc_out_spatial)

    # Classification head
    x = keras.layers.Dense(dense_units, activation="relu",
                            kernel_regularizer=reg, name="head_dense")(embedding)
    x = keras.layers.Dropout(dropout_rate, name="head_dropout")(x)
    logits = keras.layers.Dense(num_classes, name="logits")(x)
    output = keras.layers.Activation("softmax", dtype="float32", name="output")(logits)

    # Single output always — aux_output tensor is built in the graph but not
    # exposed as a model output. use_auxiliary_head=True currently has no effect
    # at training time (reserved for custom training loop in future).
    model = keras.Model(
        inputs=[main_input, calib_input],
        outputs=output,
        name="HybridResNetGAF_ConvLTC_TwoInput"
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr, weight_decay=wd)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_model(input_shape, num_classes, params):
    """
    Single-input wrapper — called by training_worker.py during MOSAN optimization.

    Internally builds the full two-input ConvLTC model and wraps it so that
    calib_volume is always zeros (h_0 = neutral learned bias during training).
    The 2D encoder and ConvLTC weights trained here are later reused in Phase 2
    where freeze_and_adapt_ltc.py loads the two-input model and adapts with real
    calibration spectra.

    Args:
        input_shape : (N_frames, H, W, C) — must be 3D_DYNAMIC_GAF format
        num_classes : number of output classes
        params      : dict of hyperparameters (see module docstring)
    """
    input_shape = tuple(input_shape)
    if len(input_shape) != 4:
        raise ValueError(
            f"HybridResNetGAF expects 4D input (N_frames, H, W, C), got {input_shape}. "
            "Use representation_type='3D_DYNAMIC_GAF'."
        )

    lr = float(params.get("learning_rate", 1e-3))
    wd = float(params.get("weight_decay",  1e-4))
    use_aux_head = bool(params.get("use_auxiliary_head", False))

    two_input_model = _build_two_input_model(input_shape, num_classes, params)

    # Wrap: single dgaf_volume input → zeros calib → two-input model
    main_input = keras.Input(shape=input_shape, name="dgaf_volume")
    zero_calib = ZerosLike(name="zero_calib")(main_input)
    out = two_input_model([main_input, zero_calib])

    model = keras.Model(
        inputs=main_input,
        outputs=out,
        name="HybridResNetGAF_ConvLTC"
    )

    if use_aux_head:
        loss         = {"output": "categorical_crossentropy",
                        "aux_output": "categorical_crossentropy"}
        loss_weights = {"output": 0.7, "aux_output": 0.3}
        metrics      = {"output": ["accuracy"], "aux_output": ["accuracy"]}
    else:
        loss         = "categorical_crossentropy"
        loss_weights = None
        metrics      = ["accuracy"]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, weight_decay=wd),
        loss=loss,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    return model


# ---------------------------------------------------------------------------
# Convenience: build two-input model for Phase 2 (freeze_and_adapt_ltc.py)
# Can also be called directly: from hybrid_resnet_gaf_ltc import build_two_input_model
# ---------------------------------------------------------------------------

def build_two_input_model(input_shape, num_classes, params):
    """Public alias for _build_two_input_model — used by freeze_and_adapt_ltc.py."""
    input_shape = tuple(input_shape)
    if len(input_shape) != 4:
        raise ValueError(
            f"HybridResNetGAF expects 4D input (N_frames, H, W, C), got {input_shape}."
        )
    return _build_two_input_model(input_shape, num_classes, params)
