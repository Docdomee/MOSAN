"""
processed_data/scripts_worker/imagenet_training_worker.py

Subprocess worker for ILSVRC ImageNet training.
Launched by cluster.tasks.run_imagenet_training_task via Celery.

Key features:
    - 8-GPU MirroredStrategy
    - bfloat16 mixed precision
    - XLA jit_compile on train step (re-enabled in this subprocess; main.py disables it globally)
    - Linear warmup LR schedule → cosine decay
    - Joint loss: CrossEntropy(cls) + lambda * SmoothL1(bbox)
    - Per-epoch checkpointing + MLFlow logging
"""
import argparse
import json
import logging
import os
import sys

# -------------------------------------------------------------------
# Project root must be on sys.path for cross-module imports
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Re-enable XLA for this worker — main.py disables it, but we want it here.
os.environ.pop("TF_XLA_FLAGS", None)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
import mlflow

logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s")
log = logging.getLogger(__name__)


# -------------------------------------------------------------------
# LR Schedule: Linear warmup → cosine decay
# -------------------------------------------------------------------
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Linear warmup for `warmup_steps`, then cosine decay to `min_lr`.

    Formula after warmup:
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * progress))
    """

    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        peak_lr = tf.cast(self.peak_lr, tf.float32)
        min_lr = tf.cast(self.min_lr, tf.float32)

        # Warmup phase
        warmup_lr = peak_lr * (step / tf.maximum(warmup_steps, 1.0))

        # Cosine decay phase
        progress = (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cosine_lr = min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + tf.cos(3.14159265 * progress))

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


# -------------------------------------------------------------------
# Loss functions
# -------------------------------------------------------------------
def smooth_l1_loss(y_true: tf.Tensor, y_pred: tf.Tensor, delta: float = 0.1) -> tf.Tensor:
    """
    Huber / Smooth L1 loss for bounding box regression.
    Robust to outlier boxes (common in annotation noise).
    """
    diff = tf.abs(y_true - y_pred)
    loss = tf.where(diff < delta, 0.5 * diff ** 2 / delta, diff - 0.5 * delta)
    return tf.reduce_mean(loss)


def build_joint_loss_fn(bbox_loss_weight: float):
    """Returns a joint classification + localization loss function."""
    cls_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction=tf.keras.losses.Reduction.NONE
    )

    def joint_loss(labels, bboxes, cls_preds, bbox_preds):
        """
        Args:
            labels: [B] int32 class labels
            bboxes: [B, 4] float32 normalized ground truth box
            cls_preds: [B, num_classes] float32 softmax probabilities
            bbox_preds: [B, 4] float32 sigmoid-bounded predicted box

        Returns:
            Scalar total loss.
        """
        cls_l = tf.reduce_mean(cls_loss_fn(labels, cls_preds))
        bbox_l = smooth_l1_loss(bboxes, bbox_preds)
        return cls_l + bbox_loss_weight * bbox_l, cls_l, bbox_l

    return joint_loss


# -------------------------------------------------------------------
# Main training loop
# -------------------------------------------------------------------
def train(args: dict, output_path: str):
    # --- 1. Distributed Strategy ---
    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync
    log.info(f"[Worker] MirroredStrategy: {num_replicas} replicas")

    # --- 2. Mixed Precision ---
    if args.get("mixed_precision", True):
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
        log.info("[Worker] Mixed precision: bfloat16 enabled")

    # --- 3. Data ---
    from imagenet.synset_utils import load_synset_mapping
    from imagenet.data_loader import ImageNetDataLoader

    synset_map = load_synset_mapping(args["synset_map"])
    global_batch = args.get("batch_size_per_gpu", 256) * num_replicas
    loader = ImageNetDataLoader(synset_map)

    train_ds = loader.build_train_dataset(
        args["dataset_dir"], global_batch, strategy=strategy
    )
    val_ds = loader.build_val_dataset(
        args["val_dir"], global_batch, strategy=strategy
    )

    # --- 4. Model ---
    from model_factory import get_imagenet_builder
    with strategy.scope():
        build_fn = get_imagenet_builder()
        model = build_fn(
            input_shape=(224, 224, 3),
            num_classes=1000,
            params={"backbone_depth": args.get("backbone_depth", 152)},
        )

        # Define strict ImageNet constants for this SPECIFIC challenge
        # (50k training images, 100k test/val images per competition rules)
        IMAGENET_TRAIN_SIZE = 50000 
        IMAGENET_VAL_SIZE = 100000

        # Estimate steps per epoch from dataset if possible
        total_epochs = args.get("total_epochs", 90)
        warmup_epochs = args.get("warmup_epochs", 5)
        
        # Use a large estimate; actual step count self-corrects via the schedule formula
        # FIX: The previous hardcoded 50,000 was a CIFAR-10 constant.
        steps_per_epoch = args.get("steps_per_epoch_estimate")
        if not steps_per_epoch:
             steps_per_epoch = IMAGENET_TRAIN_SIZE // global_batch
        
        log.info(f"[Worker] Step Calibration: {steps_per_epoch} steps/epoch (Global Batch: {global_batch})")
        
        total_steps = steps_per_epoch * total_epochs
        warmup_steps = steps_per_epoch * warmup_epochs

        # Linear LR scaling rule
        base_lr = args.get("base_lr", 0.1)
        peak_lr = base_lr * (global_batch / 256.0)

        lr_schedule = WarmupCosineDecay(peak_lr, warmup_steps, total_steps)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

        # Keep optimizer in float32 regardless of bfloat16 policy
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    joint_loss_fn = build_joint_loss_fn(args.get("bbox_loss_weight", 1.0))
    checkpoint_dir = args["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- 5. Train Step ---
    @tf.function(jit_compile=args.get("xla", True))
    def train_step(images, labels, bboxes):
        with tf.GradientTape() as tape:
            cls_preds, bbox_preds = model(images, training=True)
            total_loss, cls_l, bbox_l = joint_loss_fn(labels, bboxes, cls_preds, bbox_preds)
            scaled_loss = optimizer.get_scaled_loss(total_loss)
        scaled_grads = tape.gradient(scaled_loss, model.trainable_variables)
        grads = optimizer.get_unscaled_gradients(scaled_grads)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, cls_l, bbox_l

    @tf.function
    def val_step(images, labels):
        cls_preds, _ = model(images, training=False)
        top1 = tf.keras.metrics.sparse_top_k_categorical_accuracy(labels, cls_preds, k=1)
        top5 = tf.keras.metrics.sparse_top_k_categorical_accuracy(labels, cls_preds, k=5)
        return tf.reduce_mean(top1), tf.reduce_mean(top5)

    # --- 6. MLFlow + Training Loop ---
    try:
        from cluster.mlflow_utils import MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    except ImportError:
        mlflow.set_tracking_uri(args.get("mlflow_experiment_uri", "mlruns"))
    mlflow.set_experiment(args.get("mlflow_experiment", "ImageNet_Competition"))

    best_top1 = 0.0
    best_ckpt_path = ""
    save_every = args.get("save_every_n_epochs", 5)

    with mlflow.start_run(run_name=f"ResNet{args.get('backbone_depth', 152)}_scratch") as run:
        mlflow.log_params({
            "backbone_depth": args.get("backbone_depth", 152),
            "global_batch_size": global_batch,
            "peak_lr": peak_lr,
            "total_epochs": total_epochs,
            "mixed_precision": args.get("mixed_precision", True),
            "xla": args.get("xla", True),
        })

        for epoch in range(total_epochs):
            log.info(f"[Worker] Epoch {epoch+1}/{total_epochs}")

            # -- Training --
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_bbox_loss = 0.0
            num_batches = 0

            for batch in train_ds:
                images, labels, bboxes = batch
                loss, cls_l, bbox_l = strategy.run(train_step, args=(images, labels, bboxes))
                epoch_loss += strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None).numpy()
                epoch_cls_loss += strategy.reduce(tf.distribute.ReduceOp.MEAN, cls_l, axis=None).numpy()
                epoch_bbox_loss += strategy.reduce(tf.distribute.ReduceOp.MEAN, bbox_l, axis=None).numpy()
                num_batches += 1

            epoch_loss /= max(num_batches, 1)
            epoch_cls_loss /= max(num_batches, 1)
            epoch_bbox_loss /= max(num_batches, 1)

            # -- Validation --
            val_top1 = 0.0
            val_top5 = 0.0
            val_batches = 0
            for batch in val_ds:
                images, labels, _ = batch
                t1, t5 = strategy.run(val_step, args=(images, labels))
                val_top1 += strategy.reduce(tf.distribute.ReduceOp.MEAN, t1, axis=None).numpy()
                val_top5 += strategy.reduce(tf.distribute.ReduceOp.MEAN, t5, axis=None).numpy()
                val_batches += 1
            val_top1 /= max(val_batches, 1)
            val_top5 /= max(val_batches, 1)

            log.info(
                f"[Worker] Epoch {epoch+1} | Loss={epoch_loss:.4f} | "
                f"Top-1={val_top1:.4f} | Top-5={val_top5:.4f}"
            )

            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "train_cls_loss": epoch_cls_loss,
                "train_bbox_loss": epoch_bbox_loss,
                "val_top1": val_top1,
                "val_top5": val_top5,
                "learning_rate": float(lr_schedule(optimizer.iterations).numpy()),
            }, step=epoch)

            # -- Checkpoint --
            if (epoch + 1) % save_every == 0 or val_top1 > best_top1:
                ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}_top1_{val_top1:.4f}")
                model.save_weights(ckpt_path)
                log.info(f"[Worker] Saved checkpoint: {ckpt_path}")
                if val_top1 > best_top1:
                    best_top1 = val_top1
                    best_ckpt_path = ckpt_path
                    mlflow.log_param("best_checkpoint", ckpt_path)

        mlflow.log_metric("best_val_top1", best_top1)
        result = {
            "status": "success",
            "best_val_top1": best_top1,
            "best_checkpoint_path": best_ckpt_path,
            "mlflow_run_id": run.info.run_id,
        }

    # Write result JSON for the Celery task to read
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"[Worker] Done. Results written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_json", required=True, help="JSON string of training args")
    parser.add_argument("--output_path", required=True, help="Path for result JSON output")
    cli = parser.parse_args()
    args = json.loads(cli.args_json)
    train(args, cli.output_path)
