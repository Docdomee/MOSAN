"""
cluster/library/architectures/resnet_152_imagenet.py

ResNet (50/101/152) backbone + Detection Head for ILSVRC ImageNet.

Architecture:
    - Bottleneck residual blocks (same as He et al. 2015)
    - Depth parameterizable: 50 | 101 | 152 via params["backbone_depth"]
    - Output: two heads
        1. cls_output:  [batch, 1000] — class logits (softmax at inference)
        2. bbox_output: [batch, 4]    — normalized bbox deltas [x1,y1,x2,y2]

Mixed precision: caller should set tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
before calling build_model() for bfloat16 training.

Follows the existing architecture convention:
    build_model(input_shape, num_classes, params) -> keras.Model
"""
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# ResNet stage configs: [blocks per stage] for each depth variant
DEPTH_CONFIGS = {
    50:  [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
}

EXPANSION = 4  # Bottleneck expansion factor


def bottleneck_block(x: tf.Tensor, filters: int, stride: int = 1, name: str = "") -> tf.Tensor:
    """
    He et al. Bottleneck Block: 1×1 → 3×3 → 1×1 with optional projection shortcut.

    Args:
        x: Input tensor.
        filters: Number of filters for the 3×3 conv. Output channels = filters × EXPANSION.
        stride: Stride applied to the 3×3 conv (and shortcut projection if needed).
        name: Prefix for layer names (aids debugging).

    Returns:
        Output tensor with shape [..., filters × EXPANSION].
    """
    shortcut = x
    out_filters = filters * EXPANSION

    # Conv 1×1 (reduce)
    x = layers.Conv2D(filters, 1, use_bias=False, name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    # Conv 3×3 (spatial)
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)

    # Conv 1×1 (expand)
    x = layers.Conv2D(out_filters, 1, use_bias=False, name=f"{name}_c3")(x)
    x = layers.BatchNormalization(name=f"{name}_bn3")(x)

    # Projection shortcut (when spatial dims or channel count change)
    if stride != 1 or shortcut.shape[-1] != out_filters:
        shortcut = layers.Conv2D(out_filters, 1, strides=stride, use_bias=False, name=f"{name}_proj")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name}_relu_out")(x)
    return x


def build_stage(x: tf.Tensor, filters: int, num_blocks: int, stride: int, stage_name: str) -> tf.Tensor:
    """Builds one ResNet stage (a sequence of bottleneck blocks)."""
    # First block: apply stride + projection
    x = bottleneck_block(x, filters, stride=stride, name=f"{stage_name}_b0")
    # Remaining blocks: stride=1, no projection needed if channels match
    for i in range(1, num_blocks):
        x = bottleneck_block(x, filters, stride=1, name=f"{stage_name}_b{i}")
    return x


def build_model(
    input_shape: tuple,
    num_classes: int,
    params: dict,
) -> keras.Model:
    """
    Builds a ResNet-50/101/152 with a dual-head (classification + localization).

    Args:
        input_shape: (H, W, C) — typically (224, 224, 3).
        num_classes: Number of output classes — 1000 for ImageNet.
        params: Dict of hyperparameters:
            - backbone_depth (int): 50 | 101 | 152. Default: 152.
            - dropout_rate (float): Dropout before cls head. Default: 0.0.
            - learning_rate (float): Adam/SGD LR. Default: 0.1.
            - bbox_loss_weight (float): λ for bbox loss. NOT used here — passed to worker.

    Returns:
        keras.Model with two outputs: (cls_output, bbox_output).
        Inputs: float32 image tensor [B, H, W, 3].
    """
    depth = params.get("backbone_depth", 152)
    if depth not in DEPTH_CONFIGS:
        raise ValueError(f"[ResNet] Unsupported backbone_depth={depth}. Choose from {list(DEPTH_CONFIGS)}")

    stage_blocks = DEPTH_CONFIGS[depth]
    dropout_rate = params.get("dropout_rate", 0.0)

    inputs = keras.Input(shape=input_shape, name="image_input")

    # --- Stem ---
    # 7×7 conv, stride 2 → 3×3 max pool, stride 2 → feature map is 56×56 for 224 input
    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)

    # --- Stages ---
    # Stage 1: 64  filters, no spatial downsampling (stem pool already did it)
    x = build_stage(x, filters=64,  num_blocks=stage_blocks[0], stride=1, stage_name="s1")
    # Stage 2: 128 filters, stride 2 → 28×28
    x = build_stage(x, filters=128, num_blocks=stage_blocks[1], stride=2, stage_name="s2")
    # Stage 3: 256 filters, stride 2 → 14×14
    x = build_stage(x, filters=256, num_blocks=stage_blocks[2], stride=2, stage_name="s3")
    # Stage 4: 512 filters, stride 2 → 7×7
    feature_map = build_stage(x, filters=512, num_blocks=stage_blocks[3], stride=2, stage_name="s4")
    # feature_map shape: [B, 7, 7, 2048]

    # --- Global pooling ---
    pooled = layers.GlobalAveragePooling2D(name="gap")(feature_map)
    # pooled shape: [B, 2048]

    if dropout_rate > 0.0:
        pooled = layers.Dropout(dropout_rate, name="dropout")(pooled)

    # --- Classification Head ---
    # Cast to float32 before softmax — critical for numerical stability with bfloat16
    cls_logits = layers.Dense(num_classes, name="cls_head")(pooled)
    cls_output = layers.Activation("softmax", dtype="float32", name="cls_output")(cls_logits)

    # --- Bounding Box Regression Head ---
    # Output: [x_min, y_min, x_max, y_max] in normalized [0,1] image coordinates
    # sigmoid activation constrains output to [0,1]
    bbox_raw = layers.Dense(4, name="bbox_head")(pooled)
    bbox_output = layers.Activation("sigmoid", dtype="float32", name="bbox_output")(bbox_raw)

    model = keras.Model(
        inputs=inputs,
        outputs=[cls_output, bbox_output],
        name=f"ResNet{depth}_ImageNet_Detector",
    )

    return model
