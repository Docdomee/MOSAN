"""
processed_data/scripts_worker/imagenet_opt_worker.py

Custom Subprocess worker for Agent-driven ILSVRC ImageNet AutoML testing.
This combines the high-performance ImageNetDataLoader with the dynamic
module loading used in `training_worker.py`.

Key features:
    - Dynamic architecture loading via importlib
    - Dataset subsetting (10-20%) for fast iteration
    - Returns standardized metrics (`mean_accuracy`) for agent compatibility
"""
import argparse
import importlib.util
import json
import logging
import os
import sys
import time

# -------------------------------------------------------------------
# Project root must be on sys.path for cross-module imports
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Minimal TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import tensorflow as tf
from tensorflow.keras import backend as K

logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s")
log = logging.getLogger(__name__)


def evaluate_and_train_agent_model(args: dict, output_path: str):
    """
    Loads moving parts dynamically and trains on a subset of ImageNet.
    """
    start_time = time.time()
    
    # Extract params
    custom_architecture_file = args.get("custom_architecture_file")
    hyperparameters = args.get("hyperparameters", {})
    subset_fraction = hyperparameters.get("subset_fraction", 0.1) # Default 10%
    batch_size = hyperparameters.get("batch_size", 128)
    epochs = hyperparameters.get("epochs", 5)
    learning_rate = hyperparameters.get("learning_rate", 0.001)

    log.info(f"[AgentOpt] Loading architecture from: {custom_architecture_file}")
    
    # 1. Load Dynamic Model
    try:
        spec = importlib.util.spec_from_file_location("custom_architectures.image_net_arch", custom_architecture_file)
        architecture_module = importlib.util.module_from_spec(spec)
        sys.modules["custom_architectures.image_net_arch"] = architecture_module
        spec.loader.exec_module(architecture_module)
        build_model_func = getattr(architecture_module, "build_model")
    except Exception as e:
        log.error(f"[AgentOpt] Failed to load custom architecture: {e}")
        return _write_error(output_path, f"Architecture load error: {e}")

    # 2. Setup Data Loader
    from imagenet.synset_utils import load_synset_mapping
    from imagenet.data_loader import ImageNetDataLoader

    try:
        synset_map = load_synset_mapping(args["synset_map"])
        log.info(f"[AgentOpt] ImageNet mapped {len(synset_map)} classes. Subset={subset_fraction}")
        
        loader = ImageNetDataLoader(
            synset_map, 
            image_size=224, 
            subset_fraction=subset_fraction
        )
        
        # We don't use MirroredStrategy here by default because we might be 
        # testing hundreds of small models in parallel via Celery queue,
        # but the agent can enable it via GPU ID mapping later.
        train_ds = loader.build_train_dataset(
            dataset_dir=args["dataset_dir"], 
            global_batch_size=batch_size, 
            annotation_dir=args.get("annotation_dir")
        )
        val_ds = loader.build_val_dataset(
            dataset_dir=args["val_dir"], 
            global_batch_size=batch_size, 
            annotation_dir=args.get("annotation_dir")
        )
    except Exception as e:
         log.error(f"[AgentOpt] Failed to load dataset: {e}")
         return _write_error(output_path, f"Dataset error: {e}")

    # 3. Build Model
    try:
        K.clear_session()
        # Ensure 1000 classes and standard (224,224,3) shape
        model = build_model_func(input_shape=(224, 224, 3), num_classes=1000, **hyperparameters)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )
        log.info(f"[AgentOpt] Model compiled. Params: {model.count_params():,}")
    except Exception as e:
        log.error(f"[AgentOpt] Failed to build model: {e}")
        return _write_error(output_path, f"Model build error: {e}")

    # 4. Train
    try:
        log.info(f"[AgentOpt] Starting training for {epochs} epochs...")
        # Since it's an optimization trial, we don't need intense tracking, just the final metric
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=2 # One line per epoch
        )
        
        # We need "mean_accuracy" to align with Agent's standard tools
        val_accs = history.history.get('val_sparse_categorical_accuracy', [0.0])
        final_acc = float(val_accs[-1])
        
    except Exception as e:
        log.error(f"[AgentOpt] Training failed: {e}")
        return _write_error(output_path, f"Training error: {e}")

    # 5. Measure Inference Time
    try:
        dummy_input = tf.random.normal([1, 224, 224, 3])
        # Warmup
        model.predict(dummy_input, verbose=0)
        
        inf_start = time.time()
        for _ in range(10):
            model.predict(dummy_input, verbose=0)
        inf_time_ms = ((time.time() - inf_start) / 10.0) * 1000.0
    except Exception as e:
         log.warning(f"Inference time measurement failed: {e}")
         inf_time_ms = 0.0

    # 6. Returns Std Agent Result
    total_time = time.time() - start_time
    result = {
        "status": "completed",
        "mean_accuracy": final_acc,
        "std_accuracy": 0.0, # Not doing K-Fold on ImageNet
        "inference_time_ms": inf_time_ms,
        "params": hyperparameters,
        "message": f"Optimization trial finished in {total_time:.1f}s. Validation Acc: {final_acc:.4f}",
        "architecture_details": {"param_count": int(model.count_params())}
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
         json.dump(result, f, indent=2)
    log.info("[AgentOpt] Trial complete.")

def _write_error(output_path: str, message: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"status": "error", "message": message}, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--args_json", required=True)
    parser.add_argument("--output_path", required=True)
    cli = parser.parse_args()
    
    evaluate_and_train_agent_model(
        json.loads(cli.args_json), 
        cli.output_path
    )
