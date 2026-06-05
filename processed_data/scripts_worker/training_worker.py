import os
import sys

# --- FRAMEWORK STABILITY FIX ---
# 1. Force Protobuf to use pure-python implementation to avoid 'MessageFactory' attribute errors
# common in heterogeneous cluster environments.
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# 2. Disable oneDNN optimizations to prevent potential "double free" or registration crashes
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# -------------------------------

import argparse

# --- EARLY GPU ISOLATION FIX ---
def _early_gpu_setup():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu_id", type=int)
    args, _ = parser.parse_known_args()
    
    print(f"[Training Worker] Process started. PID: {os.getpid()}, Args: {sys.argv}")
    sys.stdout.flush()
    
    if args.gpu_id is not None:
        try:
            # Current file: processed_data/scripts_worker/training_worker.py
            # Path to root: ../..
            _proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            if _proj_root not in sys.path:
                sys.path.insert(0, _proj_root)
            
            from system_utils import logical_to_physical_gpu
            physical_gpu_id = logical_to_physical_gpu(args.gpu_id)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
            print(f"[Training Worker] Isolated to Physical GPU: {physical_gpu_id}")
        except Exception as e:
            print(f"[Training Worker] Warning during early setup: {e}")
    sys.stdout.flush()

_early_gpu_setup()
try:
    import json
    import random
    import time
    import gc
    import traceback
    import importlib.util  # Required to dynamically load --custom_architecture_file in main()
    import numpy as np
    print("[Training Worker] Basic libs imported.")
    sys.stdout.flush()
    
    import tensorflow as tf
    # Fix the NameError: define keras from tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
    print(f"[Training Worker] TensorFlow {tf.__version__} (with Keras) imported.")
    sys.stdout.flush()
    
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import LabelEncoder
    import mlflow
    import mlflow.tensorflow
    print("[Training Worker] Scikit-learn and MLFlow imported.")
    sys.stdout.flush()
except Exception as e:
    print(f"[Training Worker] CRITICAL IMPORT ERROR: {e}")
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)

# 1. Trova il percorso della cartella principale del progetto
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 2. Aggiungi la cartella principale al percorso di ricerca di Python
sys.path.append(PROJECT_ROOT)

# 3. Ora l'importazione funzionerà correttamente
from state_manager import PERSISTENT_PATHS
from state_manager import PERSISTENT_PATHS
from model_factory import MODEL_BUILDERS
import supernet  # Import the supernet module


# Disable oneDNN optimizations to prevent "double free" crashes on some setups
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Force deterministic GPU ops so fixed seeds produce identical results across runs.
# NOTE: Set to "0" because MaxPooling1D gradients crash with UnimplementedError on GPU XLA
os.environ["TF_DETERMINISTIC_OPS"] = "0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "0"

class WarmupScheduler(keras.callbacks.Callback):
    def __init__(self, warmup_epochs, initial_lr, target_lr):
        super(WarmupScheduler, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self._bypass = False  # Set in on_train_begin after model is available

    def on_train_begin(self, logs=None):
        opt_lr = self.model.optimizer.learning_rate
        if isinstance(opt_lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            self._bypass = True
            print(
                f"\n[Warmup] Model uses a LearningRateSchedule ({type(opt_lr).__name__}). "
                f"Skipping manual warmup to prevent conflict."
            )
        else:
            self._bypass = False

    def _assign_lr(self, lr):
        opt_lr = self.model.optimizer.learning_rate
        if hasattr(opt_lr, "assign"):
            opt_lr.assign(lr)
        else:
            self.model.optimizer.learning_rate = lr

    def on_epoch_begin(self, epoch, logs=None):
        if self._bypass:
            return
        if epoch < self.warmup_epochs:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (epoch / self.warmup_epochs)
            self._assign_lr(lr)
            print(f"\n[Warmup] Epoch {epoch+1}/{self.warmup_epochs}: Learning rate set to {lr:.6f}")
        elif epoch == self.warmup_epochs:
            self._assign_lr(self.target_lr)
            print(f"\n[Warmup] Warmup complete. Learning rate set to target {self.target_lr:.6f}")


def _get_history_metric(history_dict, metric_root_name):
    """
    Robustly retrieves a metric from history dict, checking common variations 
    (e.g., 'accuracy', 'acc', 'categorical_accuracy').
    """
    candidates = [
        metric_root_name,
        f"val_{metric_root_name}" if metric_root_name.startswith("val_") else None, # Should be handled by caller usually
        "acc" if metric_root_name == "accuracy" else None,
        "categorical_accuracy" if metric_root_name == "accuracy" else None,
        "spare_categorical_accuracy" if metric_root_name == "accuracy" else None,
    ]
    
    # Also handle the val_ prefix logic internally if needed, but clearer to pass exact keys
    if metric_root_name == "val_accuracy":
        candidates = ["val_accuracy", "val_acc", "val_categorical_accuracy"]
    elif metric_root_name == "accuracy":
         candidates = ["accuracy", "acc", "categorical_accuracy"]
         
    for key in candidates:
        if key and key in history_dict:
            return history_dict[key]
    return []

def main(args):
    # --- GPU SELECTION ---
    if args.gpu_id is not None:
        if "TF_CONFIG" in os.environ:
             print("[Worker] Clearing TF_CONFIG to prevent distributed strategy conflict (Single-GPU Mode requested).")
             os.environ.pop("TF_CONFIG", None)

        print(f"[Worker] GPU isolation active. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} (requested logical ID {args.gpu_id})")
    
    # --- MEMORY GROWTH FIX ---
    # Prevent "double free" / OOM by allowing memory to grow on demand
    print(f"[Worker Debug] CUDA_VISIBLE_DEVICES in env: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    try:
        gpus = tf.config.list_physical_devices('GPU')
        print(f"[Worker Debug] Physical GPUs found: {gpus}")
        if not gpus:
             print("[Worker Debug] WARNING: NO GPUS DETECTED! Falling back to CPU.")
             # raise RuntimeError("[Worker Debug] NO GPUS DETECTED! Aborting to prevent slow CPU training.")
             
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[Worker] TF Memory Growth Enabled for {len(gpus)} GPUs.")
    except Exception as e:
        print(f"[Worker Warning] Failed to set memory growth or find GPUs: {e}")
        # Re-raise explicit abort
        if "NO GPUS DETECTED" in str(e):
            raise e
            raise e
    # -------------------------

    # --- MIXED PRECISION (Memory Efficiency) ---
    if gpus:
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"[Worker] Mixed Precision enabled: {policy.compute_dtype}")
        except Exception as e:
            print(f"[Worker Warning] Failed to enable Mixed Precision: {e}")
    # -------------------------------------------

    # --- Seed Setup ---
    # Honour random_seed from hyperparameters so the agent can control replication.
    # Must parse hyperparameters first; fall back to 42 only if unspecified.
    hyperparameters = json.loads(args.params_json) if args.params_json else {}
    # Normalize augmentation alias: manifests/prompts use "augmentation", but every
    # model builder reads "use_data_augmentation". Without this the flag is a silent
    # no-op (augmentation never activates).
    if "augmentation" in hyperparameters and "use_data_augmentation" not in hyperparameters:
        hyperparameters["use_data_augmentation"] = bool(hyperparameters.pop("augmentation"))
    SEED = int(hyperparameters.get("random_seed", 42))
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    print(f"[Worker] Global seed set to {SEED} (from hyperparameters)")

    # --- CLUSTER-READY DISTRIBUTION STRATEGY ---
    # Modified to prevent freezing on local runs where TF_CONFIG might be inadvertently set or misconfigured.
    # We prioritize local MirroredStrategy for stability on single-node setups.
    if "TF_CONFIG" in os.environ and "SLURM_JOB_ID" in os.environ: # Check for SLURM to be sure it's a real cluster
        print("[Multi-Worker] TF_CONFIG and SLURM detected. Initializing MultiWorkerMirroredStrategy.")
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        # Check if TF_CONFIG exists but we are ignoring it to prevent freezes
        if "TF_CONFIG" in os.environ:
            print("[Strategy Warning] TF_CONFIG found but SLURM not detected. Ignoring to prevent local freeze. Using MirroredStrategy.")
        
        # --- STABILITY FIX: Already handled above ---
        # Memory growth must be set before GPUs are initialized. 
        # The previous block (lines 47-62) handles this. 
        # Re-setting it here is redundant and potentially dangerous.



        if gpus:
            if len(gpus) > 1:
                print(f"[Multi-GPU] Found {len(gpus)} GPUs. Initializing MirroredStrategy.")
                strategy = tf.distribute.MirroredStrategy()
            else:
                print("[Single-GPU] Found 1 GPU. Using default strategy.")
                strategy = tf.distribute.get_strategy()
        else:
            print("[CPU] No GPUs found. Using default strategy.")
            strategy = tf.distribute.get_strategy()
            
    print(f"[Strategy] Number of replicas in sync: {strategy.num_replicas_in_sync}")

    # --- BATCH SIZE SCALING (Goyal et al., 2017) ---
    # Linear Scaling Rule: LR_new = LR_base * (Global_BS / Base_BS)
    # We assume Base_BS = 32 (Standard for single GPU usually)
    global_batch_size = strategy.num_replicas_in_sync * hyperparameters.get("batch_size", 32)
    base_batch_size = 32
    scaling_factor = global_batch_size / base_batch_size
    
    print(f"[Scaling] Global Batch Size: {global_batch_size} (Replicas: {strategy.num_replicas_in_sync})")
    
    if scaling_factor > 1.0:
        base_lr = hyperparameters.get("learning_rate", 1e-3)
        scaled_lr = base_lr * scaling_factor
        #hyperparameters["learning_rate"] = scaled_lr # Update for optimizer usage
        print(f"[Scaling] Linear Scaling Rule Applied: LR {base_lr} -> {scaled_lr} (Factor: {scaling_factor:.2f})")
        print(f"[Scaling] NOTE: A 5-epoch Warmup will be applied to stabilize training.")
    else:
        print(f"[Scaling] Scaling Factor {scaling_factor:.2f} <= 1.0. Keeping original LR.")

    # Re-apply seed after strategy init (strategy init can reset TF RNG state)
    tf.random.set_seed(SEED)

    # --- MLFLOW SETUP ---
    try:
        from cluster.mlflow_utils import MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    except ImportError:
        # Fallback if cluster package is not in path (though it should be)
        try:
            from state_manager import PERSISTENT_PATHS as _PATHS
            _mlruns_path = _PATHS.get("mlruns_dir", os.path.join(PROJECT_ROOT, "processed_data", "mlruns"))
        except ImportError:
            _mlruns_path = os.path.join(PROJECT_ROOT, "processed_data", "mlruns")
        os.makedirs(_mlruns_path, exist_ok=True)
        mlflow_tracking_uri = _mlruns_path
        mlflow.set_tracking_uri(f"file:///{mlflow_tracking_uri.replace(os.sep, '/')}")
    
    # We use the experiment name passed or default
    exp_name = "Cluster_SERS_Experiments"
    mlflow.set_experiment(exp_name)
    
    # Enable Autologging for SOTA capture (graphs, params, metrics, model summary)
    # log_models=False to prevent saving full model twice (we save manually/specifically) or True if we want it.
    # We'll set it to True but be careful about storage.
    # Enable Autologging for SOTA capture (graphs, params, metrics, model summary)
    # log_models=False to prevent saving full model twice (we save manually/specifically)
    # CRITICAL FIX: Disable autologging as it causes crashes in distributed/loop environments
    # We will rely on manual logging of metrics if needed.
    mlflow.tensorflow.autolog(disable=True)


    try:
        # Determine dataset file prefix
        # CRITICAL FIX: Prioritize representation_type (e.g. 1D_CNN) over experiment_name.
        # This allows running specific trials (Exp_1) while loading standard shared datasets.
        file_prefix = hyperparameters.get("representation_type")
        
        if not file_prefix:
            if args.experiment_name:
                file_prefix = args.experiment_name
            else:
                file_prefix = "1D_CNN"
                print(f"[Worker Debug] No representation or experiment name. Defaulting to: {file_prefix}")

        print(f"[Worker] Loading data for representation: {file_prefix}")
        data_path = os.path.join(args.data_path_root, f"{file_prefix}_data.npy")
        labels_path = os.path.join(args.data_path_root, "labels.npy")
        # Load the entire dataset into memory once. tf.data will handle efficient slicing.
        # Load the primary dataset (Training Source)
        X_raw = np.load(data_path)
        
        # --- FIX: Ensure float32 for TensorFlow ---
        if X_raw.dtype == np.uint8:
            print("[Worker] Casting uint8 data to float32 and normalizing (0-1)...")
            X_raw = X_raw.astype('float32') / 255.0
        elif X_raw.dtype != np.float32:
             print(f"[Worker] Casting {X_raw.dtype} data to float32...")
             X_raw = X_raw.astype('float32')
        # ------------------------------------------
        labels_raw = np.load(labels_path, allow_pickle=True)
        le = LabelEncoder()
        labels_int_full = le.fit_transform(labels_raw)
        num_classes = len(le.classes_)
        y_raw = keras.utils.to_categorical(labels_int_full, num_classes=num_classes)
        
        # --- DIAGNOSTIC PRINT (USER REQUESTED CHECK) ---
        print(f"[Worker Diagnostic] LOADED X_raw SHAPE: {X_raw.shape}")
        print(f"[Worker Diagnostic] n_dims: {X_raw.ndim}")
        # -----------------------------------------------

        # --- TEST SET STRATEGY (FIXED) ---
        X_test_heldout = None
        y_test_heldout = None
        
        # PRIORITIZE EXTERNAL FOLDER IF PASSED
        test_load_root = args.test_data_folder if args.test_data_folder else args.data_path_root
        
        dynamic_test_path = os.path.join(test_load_root, f"{file_prefix}_X_test_heldout.npy")
        dynamic_test_labels = os.path.join(test_load_root, "labels_test_heldout.npy")
        
        if os.path.exists(dynamic_test_path) and os.path.exists(dynamic_test_labels):
             print(f"[Worker] Discovered explicitly generated 'X_test_heldout.npy' in {test_load_root}.")
             try:
                 X_test_heldout = np.load(dynamic_test_path)
                 
                 # --- FIX: Cast External Test Data ---
                 if X_test_heldout.dtype == np.uint8:
                    print("[Worker] Casting external test uint8 data to float32 and normalizing...")
                    X_test_heldout = X_test_heldout.astype('float32') / 255.0
                 elif X_test_heldout.dtype != np.float32:
                    X_test_heldout = X_test_heldout.astype('float32')
                 # ------------------------------------
                 labels_test_raw = np.load(dynamic_test_labels, allow_pickle=True)
                 y_test_heldout = keras.utils.to_categorical(le.transform(labels_test_raw), num_classes=num_classes)
                 
                 # --- BETTER PROOF: Load signature ---
                 test_mean = np.mean(X_test_heldout)
                 print(f"[Worker Proof] Loaded test set from {test_load_root}. Shape: {X_test_heldout.shape}, Content Mean: {test_mean:.6f}")
                 
                 X_train_full = X_raw
                 y_train_full = y_raw
                 labels_int_train = labels_int_full
                 setattr(args, "has_dynamic_test_folder", True)
             except Exception as e:
                 print(f"[Worker Error] Dynamic test set load failed: {e}. Falling back to internal split.")
                 X_train_full, X_test_heldout, y_train_full, y_test_heldout = train_test_split(
                    X_raw, y_raw, test_size=0.2, stratify=labels_int_full, random_state=SEED
                 )
                 labels_int_train = np.argmax(y_train_full, axis=1)
                 setattr(args, "has_dynamic_test_folder", False)
        else:
             print("[Worker] No external test set provided. Performing internal 80/20 Stratified Split.")
             X_train_full, X_test_heldout, y_train_full, y_test_heldout = train_test_split(
                X_raw, y_raw, test_size=0.2, stratify=labels_int_full, random_state=SEED
             )
             labels_int_train = np.argmax(y_train_full, axis=1)
             setattr(args, "has_dynamic_test_folder", False)
             
        # Assign to variables used by CV loop
        X = X_train_full
        y = y_train_full
        labels_int = labels_int_train
        
        print(f"[Worker Data] Train Shape: {X.shape}, Held-Out Test Shape: {X_test_heldout.shape}")

        # --- DATA INTEGRITY CHECK (MD5) ---
        import hashlib
        def get_array_checksum(arr):
             # Fast hash of first 1MB or full array if small
             data_slice = arr.ravel()[:250000] # Approx 1MB for float32
             return hashlib.md5(data_slice.tobytes()).hexdigest()
             
        print(f"[Data Integrity] X_raw Checksum (Partial): {get_array_checksum(X_raw)}")
        print(f"[Data Integrity] labels Checksum (Partial): {get_array_checksum(labels_raw)}")
        # ----------------------------------
        
        # --- tf.data.Dataset Configuration ---
        per_replica_batch_size = hyperparameters.get("batch_size", 32)
        global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
        print(f"[tf.data] Per-replica batch size: {per_replica_batch_size}, Global batch size: {global_batch_size}")
        print(f"[Worker Debug] num_replicas_in_sync: {strategy.num_replicas_in_sync}")
        if global_batch_size * 5 > len(X):
             print("[Worker Warning] Global batch size is large relative to dataset size. Convergence might be unstable.")

        # --- PRE-DISTRIBUTE HELD-OUT SET FOR FAST FOLD EVALUATION ---
        held_out_dist_ds = None
        held_out_steps = 0
        if X_test_heldout is not None and len(X_test_heldout) > 0:
            held_out_ds = tf.data.Dataset.from_tensor_slices((X_test_heldout, y_test_heldout))
            held_out_ds = held_out_ds.batch(global_batch_size).prefetch(tf.data.AUTOTUNE)
            held_out_dist_ds = strategy.experimental_distribute_dataset(held_out_ds)
            held_out_steps = (len(X_test_heldout) + global_batch_size - 1) // global_batch_size
            print(f"[Worker] Held-out test set distributed for multi-fold evaluation.")

    except Exception:
        result = {"status": "error", "message": f"Data loading failed: {traceback.format_exc()}"}
        with open(args.output_path, "w") as f:
            json.dump(result, f)
        return

    # --- Dispatch based on Mode ---
    if args.mode == "supernet":
        try:
            with mlflow.start_run(run_name=f"Supernet_{args.dataset_id}"):
                mlflow.log_params(hyperparameters)
                mlflow.log_param("mode", "supernet")
                mlflow.log_param("dataset_id", args.dataset_id)
                train_supernet(args, strategy, X, y, num_classes, hyperparameters)
        except Exception:
             result = {"status": "error", "message": f"Supernet training failed: {traceback.format_exc()}"}
             with open(args.output_path, "w") as f:
                json.dump(result, f)
        return

    # --- PRE-FLIGHT ARCHITECTURE LOAD ---
    # Load the module outside strategy.scope() and CV loop to prevent cluster synchronization freezes on FileNotFoundError
    model_builder = None
    custom_module = None
    if args.custom_architecture_file:
        filepath = os.path.join(PERSISTENT_PATHS["custom_architectures_dir"], args.custom_architecture_file)
        if not os.path.exists(filepath) and not filepath.endswith(".py"):
            filepath += ".py"
            
        if not os.path.exists(filepath):
            result = {"status": "error", "message": f"Custom architecture file not found: {filepath}"}
            with open(args.output_path, "w") as f:
                json.dump(result, f)
            print(f"[Worker Error] Architecture file missing: {filepath}. Exiting before Strategy init to avoid freeze.")
            return

        # Ghost-execution guard. Three layers, all needed:
        #  (1) sys.modules.pop  — evict any in-process cached module object
        #  (2) invalidate_caches — clear sys.path_importer_cache (path finders)
        #  (3) delete __pycache__/<mod>.* — Python's SourceFileLoader checks the
        #      .pyc mtime against source mtime; when write_architecture_file rewrites
        #      the source within the same FS-clock second (common on cluster NFS),
        #      mtimes match and the stale .pyc is served. Removing the .pyc forces
        #      re-compile from source on the very next import.
        import glob as _glob
        _mod_name = args.custom_architecture_file.replace(".py", "")
        sys.modules.pop(_mod_name, None)
        importlib.invalidate_caches()
        _pycache_dir = os.path.join(os.path.dirname(filepath), "__pycache__")
        if os.path.isdir(_pycache_dir):
            for _pyc in _glob.glob(os.path.join(_pycache_dir, _mod_name + ".*")):
                try:
                    os.remove(_pyc)
                except OSError:
                    pass
        spec = importlib.util.spec_from_file_location(_mod_name, filepath)
        if spec is None or spec.loader is None:
            result = {"status": "error", "message": f"Could not load spec/loader for {filepath}"}
            with open(args.output_path, "w") as f:
                json.dump(result, f)
            print(f"[Worker Error] Could not load spec: {filepath}")
            return
            
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        model_builder = custom_module.build_model
    else:
        model_builder = MODEL_BUILDERS.get(args.experiment_name) or MODEL_BUILDERS.get(hyperparameters.get("representation_type"))

    if not model_builder:
        result = {"status": "error", "message": f"Model builder for '{args.experiment_name}' or type '{hyperparameters.get('representation_type')}' not found."}
        with open(args.output_path, "w") as f:
            json.dump(result, f)
        print(f"[Worker Error] Model builder missing.")
        return
    # ------------------------------------

    fold_accuracies = []
    fold_train_accuracies = []
    fold_val_losses = []
    fold_overfitting_scores = []
    fold_learning_speeds = []
    fold_convergence_stabilities = []
    fold_training_epochs = []
    fold_heldout_accuracies = [] # Track test scores per fold
    checkpoint_history = []  # Validation metrics logged at each checkpoint epoch
    checkpoint_every = int(hyperparameters.get("checkpoint_every_n_epochs", 0))

    class PeriodicCheckpointCallback(keras.callbacks.Callback):
        """Logga le metriche di validazione ogni N epoche nel result JSON.
        Legge i valori già calcolati da model.fit — nessuna dipendenza dal tipo di dataset."""
        def __init__(self, every_n, fold_id, history_list):
            super().__init__()
            self.every_n = every_n
            self.fold_id = fold_id
            self.history = history_list

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.every_n != 0:
                return
            logs = logs or {}
            entry = {"fold": self.fold_id, "epoch": epoch + 1}
            for key in ("val_loss", "val_accuracy", "loss", "accuracy"):
                if key in logs:
                    entry[key] = float(logs[key])
            self.history.append(entry)
            val_acc = logs.get("val_accuracy", logs.get("val_acc", "n/a"))
            print(f"[Checkpoint] fold={self.fold_id} epoch={epoch+1} val_acc={val_acc}")
            mlflow.log_metric(f"ckpt_val_acc_fold{self.fold_id}",
                              float(val_acc) if isinstance(val_acc, (int, float)) else 0,
                              step=epoch + 1)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    cv_splits = list(skf.split(X, labels_int))

    # Start a parent run for the cross-validation
    with mlflow.start_run(run_name=f"CV_{args.experiment_name}", description="Cross-Validation Run") as parent_run:
        mlflow.log_params(hyperparameters)
        mlflow.log_param("experiment_name", args.experiment_name)
        
        for fold, (train_index, test_index) in enumerate(cv_splits):
            print(f"--- Starting Fold {fold + 1}/{len(cv_splits)} ---")
            
            # Nested run for each fold
            with mlflow.start_run(run_name=f"Fold_{fold+1}", nested=True):
                model = None
                try:
                    # --- Create tf.data.Dataset for current fold ---
                    # We ALWAYS evaluate on the true internal K-Fold structure
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

                    # Apply shuffling, batching, and prefetching
                    # Apply shuffling, batching, and prefetching
                    # DETERMINISM FIX: Explicit seed for shuffling
                    train_dataset = train_dataset.shuffle(buffer_size=len(X_train), seed=SEED)
                    train_dataset = train_dataset.batch(global_batch_size).prefetch(tf.data.AUTOTUNE).repeat()
                    test_dataset = test_dataset.batch(global_batch_size).prefetch(tf.data.AUTOTUNE).repeat()
                    
                    # Distribute the datasets among replicas
                    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
                    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)


                    with strategy.scope():
                        model = model_builder(X_train.shape[1:], num_classes, hyperparameters)
                        
                        # PERFORMANCE BOOST: Enable XLA for custom architectures or if requested
                        # Custom/NAS architectures have many small ops (fragmented graph) that benefit MASSIVELY from XLA fusion.
                        if args.custom_architecture_file or hyperparameters.get("use_xla", False):
                            try:
                                print("[Worker] Enabling XLA (jit_compile=True) for optimized execution...")
                                # We must re-compile to enable JIT. We try to preserve existing configuration.
                                # Safe metric retrieval:
                                current_metrics = ["accuracy"]
                                if hasattr(model, 'metrics_names'):
                                    # Fallback to names if objects fail, though re-compiling with names is safer for strings
                                    current_metrics = ["accuracy"] 

                                model.compile(
                                    optimizer=model.optimizer,
                                    loss=model.loss,
                                    metrics=current_metrics,
                                    jit_compile=True
                                )
                            except Exception as e:
                                print(f"[Worker Warning] XLA Enablement Failed: {e}. continuing without XLA.")

                    # --- Custom Training Loop Support ---
                    if custom_module and hasattr(custom_module, "train_model"):
                        print(f"[Custom Training] Using 'train_model' from {args.custom_architecture_file}")
                        # The custom train_model must return a History object or dict-like history
                        history = custom_module.train_model(
                            model, 
                            train_dist_dataset, 
                            test_dist_dataset, 
                            epochs=100, 
                            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
                            verbose=0
                        )
                    else:
                        # Standard Training Loop
                        # FIX: Ideally the model_builder should return a compiled model, but if it doesn't (like GENOTYPE models),
                        # we MUST compile it here with standard defaults or hyperparameters.
                        if not getattr(model, "optimizer", None):
                            opt_name = hyperparameters.get("optimizer", "adam").lower()
                            lr = hyperparameters.get("learning_rate", 1e-3)
                            
                            print(f"[Worker] Compiling model with {opt_name} (lr={lr})")
                            
                            if opt_name == "sgd":
                                optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
                            elif opt_name == "adamw":
                                # AdamW requires weight decay, usually passed separately or defaulted
                                optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)        
                            else:
                                optimizer = keras.optimizers.Adam(learning_rate=lr)

                            model.compile(
                                optimizer=optimizer,
                                loss="categorical_crossentropy",
                                metrics=["accuracy"]
                            )

                        # Default patience increased for stability
                        patience = hyperparameters.get("patience", 15)
                        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

                        current_callbacks = [early_stopping]

                        if checkpoint_every > 0:
                            current_callbacks.append(
                                PeriodicCheckpointCallback(checkpoint_every, fold + 1, checkpoint_history)
                            )
                            print(f"[Worker] PeriodicCheckpoint every {checkpoint_every} epochs enabled (fold {fold+1})")

                        # --- CHECKPOINTING FIX ---
                        if hyperparameters.get("save_model", False):
                            checkpoint_dir = os.path.join(PERSISTENT_PATHS["models_dir"], "checkpoints")
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            fold_ckpt_name = f"{args.experiment_name}_fold{fold+1}.keras"
                            checkpoint_path = os.path.join(checkpoint_dir, fold_ckpt_name)
                            
                            checkpoint_cb = keras.callbacks.ModelCheckpoint(
                                filepath=checkpoint_path,
                                monitor="val_loss",
                                save_best_only=True,
                                verbose=1
                            )
                            current_callbacks.append(checkpoint_cb)
                            print(f"[Worker] ModelCheckpoint attached for Fold {fold+1}: {checkpoint_path}")
                        # -------------------------

                        # Add Warmup Callback if Valid
                        if scaling_factor > 1.0:
                             # Start from small LR (e.g. 1/scaling_factor of base) or 0
                             base_lr = hyperparameters.get("learning_rate", 1e-3)
                             start_lr = base_lr / scaling_factor 
                             warmup_cb = WarmupScheduler(warmup_epochs=5, initial_lr=start_lr, target_lr=base_lr)
                             current_callbacks.append(warmup_cb)
                             print(f"[Worker] Warmup Scheduler attached (0 -> {base_lr:.6f} over 5 epochs)")

                        # Fix for DistributedDataset: Must provide steps
                        steps_per_epoch = (len(X_train) + global_batch_size - 1) // global_batch_size
                        validation_steps = (len(X_test) + global_batch_size - 1) // global_batch_size

                        history = model.fit(
                            train_dist_dataset,
                            epochs=hyperparameters.get("epochs", 100),
                            steps_per_epoch=steps_per_epoch,
                            validation_data=test_dist_dataset,
                            validation_steps=validation_steps,
                            callbacks=current_callbacks,
                            verbose=0,
                        )
                        
                        # --- DIAGNOSTIC PRINT ---
                        print(f"[Worker Diagnostic] Fold {fold+1} model.fit finished.")
                        print(f"[Worker Diagnostic] History keys: {list(history.history.keys())}")
                        if "val_accuracy" in history.history:
                            print(f"[Worker Diagnostic] val_accuracy length (epochs ran): {len(history.history['val_accuracy'])}")
                        # ------------------------


                    # --- Evaluation on the distributed test set ---
                    eval_result = model.evaluate(test_dist_dataset, steps=validation_steps, verbose=0)
                    accuracy = eval_result[1] # The metric is at index 1
                    fold_accuracies.append(accuracy)
                    
                    # --- NEW: HELD-OUT EVALUATION PER FOLD ---
                    if held_out_dist_ds is not None:
                        heldout_eval = model.evaluate(held_out_dist_ds, steps=held_out_steps, verbose=0)
                        heldout_acc = heldout_eval[1]
                        fold_heldout_accuracies.append(heldout_acc)
                        print(f"--- Fold {fold + 1}/3 - CV Acc: {accuracy:.4f} | Held-out Acc: {heldout_acc:.4f} ---")
                        mlflow.log_metric("held_out_accuracy", heldout_acc, step=fold)
                    else:
                        print(f"--- Fold {fold + 1}/3 Completed. Accuracy: {accuracy:.4f} ---")
                    
                    # LOGGING CRITICAL FOR DIAGNOSIS
                    print(f"[Fold {fold+1} Debug] History Keys: {history.history.keys()}")
                    if "loss" in history.history:
                         print(f"[Fold {fold+1} Debug] Final Training Loss: {history.history['loss'][-1]}")
                    if "val_loss" in history.history:
                         print(f"[Fold {fold+1} Debug] Final Val Loss: {history.history['val_loss'][-1]}")

                    val_acc = _get_history_metric(history.history, "val_accuracy")
                    train_acc = _get_history_metric(history.history, "accuracy")
                    
                    if not train_acc and val_acc:
                        # Fallback if metric was logged under a slightly different name
                        for key in history.history.keys():
                            if "acc" in key.lower() and "val" not in key.lower():
                                train_acc = history.history[key]
                                break
                    
                    if train_acc:
                        fold_train_accuracies.append(train_acc[-1])
                    val_loss = history.history.get("val_loss", [])
                    epochs_ran = len(val_acc) if val_acc else 0

                    fold_training_epochs.append(epochs_ran)
                    if val_loss:
                        fold_val_losses.append(min(val_loss))

                    if epochs_ran >= 5:
                        overfitting = np.mean(np.array(train_acc[-5:]) - np.array(val_acc[-5:]))
                        fold_overfitting_scores.append(overfitting)
                        stability = np.std(val_acc[-5:])
                        fold_convergence_stabilities.append(stability)

                    if epochs_ran >= 10:
                        slope = np.polyfit(range(10), val_acc[:10], 1)[0]
                        fold_learning_speeds.append(slope)

                except Exception:
                    result = {"status": "error", "message": f"Training error in Fold {fold + 1}: {traceback.format_exc()}"}
                    with open(args.output_path, "w") as f:
                        json.dump(result, f)
                    return
                finally:
                    if model:
                        del model
                    K.clear_session()
                    gc.collect()

        # --- LOG AGGREGATE METRICS TO MLFLOW (Parent Run: CV_Experiment) ---
        # Calculated after the loop, but must be inside the 'with mlflow.start_run(...)' block
        if fold_accuracies:
             mlflow.log_metric("mean_cv_accuracy", np.mean(fold_accuracies))
             mlflow.log_metric("std_cv_accuracy", np.std(fold_accuracies))
        if fold_train_accuracies:
             mlflow.log_metric("mean_cv_train_accuracy", np.mean(fold_train_accuracies))
        if fold_overfitting_scores:
             mlflow.log_metric("mean_overfitting_score", np.mean(fold_overfitting_scores))
        if fold_learning_speeds:
             mlflow.log_metric("mean_learning_speed", np.mean(fold_learning_speeds))
        if fold_convergence_stabilities:
             mlflow.log_metric("mean_convergence_stability", np.mean(fold_convergence_stabilities))
        if fold_training_epochs:
             mlflow.log_metric("mean_training_epochs", np.mean(fold_training_epochs))
        
        if fold_heldout_accuracies:
             mlflow.log_metric("mean_cv_heldout_accuracy", np.mean(fold_heldout_accuracies))
             print(f"[MLFlow] Mean Held-out Accuracy across folds: {np.mean(fold_heldout_accuracies):.4f}")
        
        print("[MLFlow] Aggregate CV metrics logged.")
        # -------------------------------------------------------------------

    model_path = None
    held_out_test_accuracy = 0.0
    ho_f1_macro = ho_f1_weighted = ho_mcc = None
    shifted_test_accuracy = None
    shifted_f1_macro = shifted_f1_weighted = shifted_mcc = None
    evaluation_error = None
    try:
        # Create final dataset for the full training run
        full_dataset = tf.data.Dataset.from_tensor_slices((X, y))
        full_dataset = full_dataset.shuffle(buffer_size=len(X), seed=SEED)
        full_dataset = full_dataset.batch(global_batch_size).prefetch(tf.data.AUTOTUNE).repeat()
        full_dist_dataset = strategy.experimental_distribute_dataset(full_dataset)

        with strategy.scope():
            final_model = model_builder(X.shape[1:], num_classes, hyperparameters)

            # PERFORMANCE BOOST: Enable XLA for final model too
            if args.custom_architecture_file or hyperparameters.get("use_xla", False):
                try:
                    print("[Worker Final] Enabling XLA (jit_compile=True)...")
                    final_model.compile(
                         optimizer=final_model.optimizer,
                         loss=final_model.loss,
                         metrics=["accuracy"], # Safe fallback
                         jit_compile=True
                    )
                except Exception as e:
                    print(f"[Worker Warning] XLA Final Enablement Failed: {e}")

            # Ensure the final model is compiled regardless of XLA outcome.
            # model_builder for custom architectures may return an uncompiled Functional model.
            # Use getattr(...,"optimizer") — Functional models in TF>=2.12 have no `.compiled`
            # attribute, so the old check raised AttributeError and zeroed held_out_test_accuracy.
            if getattr(final_model, "optimizer", None) is None:
                print("[Worker Final] Model not compiled after XLA attempt. Compiling with standard settings...")
                final_model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=hyperparameters.get("learning_rate", 1e-3)),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]
                )

        # Train on the full distributed dataset
        # We start a separate run or continue the parent? 
        # Typically "Final Model" is the most important, so we might want it in the parent or a sibling run.
        # Let's verify if we are still in parent context? Yes, 'with mlflow.start_run' above wraps the loop. 
        # Wait, the indentation above ended the loop but the 'with' block needs to be closed.
        # The previous replacement closed the loop indentation but not the 'with'. 
        # Actually my replacement for standard training inserted 'with ... as parent_run:' 
        # so everything indented below it is in the run.
        # But this section'model_path = None' is OUTSIDE the loop in the original code.
        # So I need to handle the indentation correctly or start a new run for Final Training.
        
        # SERS SOTA: Final training is what we deploy, so we log it prominently.
        with mlflow.start_run(run_name=f"Final_Model_{args.experiment_name}", nested=True):
            mlflow.log_params(hyperparameters)
            mlflow.log_param("training_type", "full_dataset_retraining")
            
            # Use EarlyStopping here too (monitoring training loss since there is no validation set for final model)
            final_early_stopping = keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
            
            final_callbacks = [final_early_stopping]

            # --- CHECKPOINTING FIX (Final Training) ---
            if hyperparameters.get("save_model", False):
                checkpoint_dir = os.path.join(PERSISTENT_PATHS["models_dir"], "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                final_ckpt_name = f"{args.experiment_name}_final_best.keras"
                checkpoint_path = os.path.join(checkpoint_dir, final_ckpt_name)
                
                checkpoint_cb = keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor="loss",
                    save_best_only=True,
                    verbose=1
                )
                final_callbacks.append(checkpoint_cb)
                print(f"[Worker Final] ModelCheckpoint attached: {checkpoint_path}")
            # ------------------------------------------

            # Calculate steps for final training
            final_steps_per_epoch = (len(X) + global_batch_size - 1) // global_batch_size
            final_model.fit(full_dist_dataset, epochs=int(np.mean(fold_training_epochs)), steps_per_epoch=final_steps_per_epoch, callbacks=final_callbacks, verbose=0)
            
            # Check if model saving is requested (Defaults to False if not specified, 
            # unless running as 'Final_Model' in which case valid tool call should specify it).
            # Actually, per user request, we default to FALSE to save space.
            if hyperparameters.get("save_model", False):
                model_dir = PERSISTENT_PATHS["models_dir"]
                os.makedirs(model_dir, exist_ok=True)
                # Use a unique name for the model
                model_filename = f"{args.experiment_name}_{os.path.basename(args.output_path).replace('.json', '')}.keras"
                model_path = os.path.join(model_dir, model_filename)
                final_model.save(model_path)
                print(f"Final model saved to {model_path}")
                
                try:
                    # Log the model artifact manually too if autolog didn't catch it nicely (autolog usually does)
                    mlflow.log_artifact(model_path, artifact_path="model_files")
                except Exception as e:
                    print(f"[Worker Warning] Failed to log artifact to MLflow: {e}")
            else:
                print("[Worker] 'save_model' is False or missing. Model NOT saved to disk.")
                model_path = "Model not saved (save_model=False)"
            
            # --- EVALUATION ON HELD-OUT TEST SET ---
            held_out_test_accuracy = 0.0
            evaluation_error = None  # Initialize error tracking

            if X_test_heldout is not None and len(X_test_heldout) > 0:
                print(f"[Final Model] Evaluating on Held-Out Test Set ({len(X_test_heldout)} samples)...")
                
                # Check for label consistency
                # if num_classes != y_test_heldout.shape[1]:
                #     print(f"[Final Model Warning] Mismatch in classes: Model {num_classes} vs Data {y_test_heldout.shape[1]}")

                # Calculate adequate steps for distributed evaluation
                test_steps = (len(X_test_heldout) + global_batch_size - 1) // global_batch_size
                
                # Create dataset
                final_test_ds = tf.data.Dataset.from_tensor_slices((X_test_heldout, y_test_heldout))
                final_test_ds = final_test_ds.batch(global_batch_size).prefetch(tf.data.AUTOTUNE)
                
                try:
                    # Pass steps explicitly to avoid distributed strategy issues
                    eval_res = final_model.evaluate(final_test_ds, steps=test_steps, verbose=0)
                    
                    if isinstance(eval_res, list):
                         # Usually [loss, accuracy]
                         held_out_test_accuracy = eval_res[1] 
                    elif isinstance(eval_res, dict):
                         # If return_dict=True
                         held_out_test_accuracy = eval_res.get("accuracy", 0.0)
                    else:
                         # Scalar (loss only?) or single metric
                         held_out_test_accuracy = eval_res 
                    
                    mlflow.log_metric("held_out_test_accuracy", held_out_test_accuracy)
                    print(f"[Final Model] Test Set Accuracy: {held_out_test_accuracy:.4f}")

                except Exception as e:
                    evaluation_error = str(e)
                    print(f"[Final Model] Test Set Evaluation Failed: {e}")
                    traceback.print_exc()
            else:
                 print("[Final Model] No Held-Out Test Data available for evaluation.")

            # F1 / MCC on held-out test — requires raw predictions via model.predict()
            ho_f1_macro = ho_f1_weighted = ho_mcc = None
            if X_test_heldout is not None and len(X_test_heldout) > 0:
                try:
                    from sklearn.metrics import f1_score as _f1sk, matthews_corrcoef as _mccsk
                    _y_true_int = np.argmax(y_test_heldout, axis=1)
                    _y_pred_int = final_model.predict(
                        X_test_heldout, batch_size=256, verbose=0).argmax(axis=1)
                    ho_f1_macro    = float(_f1sk(_y_true_int, _y_pred_int, average="macro",    zero_division=0))
                    ho_f1_weighted = float(_f1sk(_y_true_int, _y_pred_int, average="weighted", zero_division=0))
                    ho_mcc         = float(_mccsk(_y_true_int, _y_pred_int))
                    print(f"[Final Model] HO F1-macro={ho_f1_macro:.4f}  MCC={ho_mcc:.4f}")
                except Exception as _e:
                    print(f"[Final Model] F1/MCC computation skipped: {_e}")

            # --- SHIFTED TEST EVALUATION (diagnostic) ---
            # If the dataset has a shifted_test split (different distribution
            # from training, e.g. aged or clinical), evaluate the final model
            # on it as a DIAGNOSTIC metric. This is NOT used for EarlyStopping
            # — it just reports the distribution-shift gap so the agent knows
            # whether to escalate to run_training_trial_with_finetune.
            shifted_test_accuracy = None
            try:
                shifted_x_path = os.path.join(
                    args.data_path_root,
                    f"{hyperparameters.get('representation_type', '')}_X_shifted_test.npy")
                shifted_y_path = os.path.join(args.data_path_root, "labels_shifted_test.npy")
                if os.path.exists(shifted_x_path) and os.path.exists(shifted_y_path):
                    X_shifted = np.load(shifted_x_path)
                    if X_shifted.dtype == np.uint8:
                        X_shifted = X_shifted.astype("float32") / 255.0
                    elif X_shifted.dtype != np.float32:
                        X_shifted = X_shifted.astype("float32")
                    labels_shifted_raw = np.load(shifted_y_path, allow_pickle=True)
                    y_shifted = keras.utils.to_categorical(
                        le.transform(labels_shifted_raw), num_classes=num_classes)
                    shifted_steps = (len(X_shifted) + global_batch_size - 1) // global_batch_size
                    shifted_ds = tf.data.Dataset.from_tensor_slices((X_shifted, y_shifted))
                    shifted_ds = shifted_ds.batch(global_batch_size).prefetch(tf.data.AUTOTUNE)
                    sh_res = final_model.evaluate(shifted_ds, steps=shifted_steps, verbose=0)
                    if isinstance(sh_res, list):
                        shifted_test_accuracy = float(sh_res[1])
                    elif isinstance(sh_res, dict):
                        shifted_test_accuracy = float(sh_res.get("accuracy", 0.0))
                    else:
                        shifted_test_accuracy = float(sh_res)
                    mlflow.log_metric("shifted_test_accuracy", shifted_test_accuracy)
                    print(f"[Final Model] Shifted Test Accuracy (diagnostic): {shifted_test_accuracy:.4f}")
                    gap = held_out_test_accuracy - shifted_test_accuracy
                    print(f"[Final Model] Distribution-shift gap (HO - shifted): {gap:+.4f}")

                    # F1 / MCC on shifted test
                    shifted_f1_macro = shifted_f1_weighted = shifted_mcc = None
                    try:
                        from sklearn.metrics import f1_score as _f1sk2, matthews_corrcoef as _mccsk2
                        _y_shifted_int_sk = le.transform(labels_shifted_raw)
                        _y_shifted_pred   = final_model.predict(
                            X_shifted, batch_size=256, verbose=0).argmax(axis=1)
                        shifted_f1_macro    = float(_f1sk2(_y_shifted_int_sk, _y_shifted_pred, average="macro",    zero_division=0))
                        shifted_f1_weighted = float(_f1sk2(_y_shifted_int_sk, _y_shifted_pred, average="weighted", zero_division=0))
                        shifted_mcc         = float(_mccsk2(_y_shifted_int_sk, _y_shifted_pred))
                        print(f"[Final Model] Shifted F1-macro={shifted_f1_macro:.4f}  MCC={shifted_mcc:.4f}")
                    except Exception as _e2:
                        print(f"[Final Model] Shifted F1/MCC computation skipped: {_e2}")
                else:
                    shifted_f1_macro = shifted_f1_weighted = shifted_mcc = None
                    print("[Final Model] No shifted test set found (this is fine if dataset has no distribution shift).")
            except Exception as _sh_e:
                print(f"[Final Model] Shifted test evaluation skipped due to error: {_sh_e}")
                shifted_test_accuracy = None
                shifted_f1_macro = shifted_f1_weighted = shifted_mcc = None

            # --- PROFILING ---
            # Create a non-distributed dataset for accurate inference timing
            profile_ds = tf.data.Dataset.from_tensor_slices((X, y)).batch(global_batch_size)
            profiling_results = profile_model(final_model, profile_ds)
            
            mlflow.log_metrics(profiling_results)

        
    except Exception as _final_exc:
        evaluation_error = f"{type(_final_exc).__name__}: {_final_exc}"
        print(f"[Worker CRASH] Failed to train or save the final model. Traceback below:")
        traceback.print_exc()
        profiling_results = {}


    # Build the full effective hyperparameter dict: agent-supplied values + any defaults
    # applied during training. This ensures the results_log captures exactly what was trained.
    effective_hyperparameters = {
        "num_conv_layers": hyperparameters.get("num_conv_layers", 2),
        "num_layers": hyperparameters.get("num_conv_layers", 2),  # alias so analytics can query either key
        "filters": hyperparameters.get("filters", 32),
        "kernel_size": hyperparameters.get("kernel_size", 3),
        "dense_units": hyperparameters.get("dense_units", 128),
        "dropout_rate": hyperparameters.get("dropout_rate", 0.5),
        "learning_rate": hyperparameters.get("learning_rate", 1e-3),
        "batch_size": hyperparameters.get("batch_size", 32),
        "epochs": hyperparameters.get("epochs", 50),
        "patience": hyperparameters.get("patience", 15),
        "batch_norm": hyperparameters.get("batch_norm", 0.0),
        "weight_decay": hyperparameters.get("weight_decay", 0.0),
        "use_residual": hyperparameters.get("use_residual", False),
        "use_l1_regularization": hyperparameters.get("use_l1_regularization", False),
        "use_l2_regularization": hyperparameters.get("use_l2_regularization", False),
        "use_data_augmentation": hyperparameters.get("use_data_augmentation", False),
    }
    # Include any extra keys the agent passed that are not in the canonical list
    for k, v in hyperparameters.items():
        if k not in effective_hyperparameters:
            effective_hyperparameters[k] = v

    final_result = {
        "status": "completed",
        "mean_accuracy": np.mean(fold_accuracies) if fold_accuracies else 0,
        "mean_train_accuracy": np.mean(fold_train_accuracies) if fold_train_accuracies else 0,
        "mean_heldout_accuracy": np.mean(fold_heldout_accuracies) if fold_heldout_accuracies else held_out_test_accuracy,
        "held_out_test_accuracy": held_out_test_accuracy,
        "held_out_f1_macro":    ho_f1_macro,
        "held_out_f1_weighted": ho_f1_weighted,
        "held_out_mcc":         ho_mcc,
        "shifted_test_accuracy": shifted_test_accuracy,  # None when dataset has no shifted split
        "shifted_f1_macro":     shifted_f1_macro,
        "shifted_f1_weighted":  shifted_f1_weighted,
        "shifted_mcc":          shifted_mcc,
        "evaluation_error": evaluation_error,
        "std_accuracy": np.std(fold_accuracies) if fold_accuracies else 0,
        "mean_final_val_loss": np.mean(fold_val_losses) if fold_val_losses else 0,
        "mean_training_epochs": np.mean(fold_training_epochs) if fold_training_epochs else 0,
        "mean_overfitting_score": np.mean(fold_overfitting_scores) if fold_overfitting_scores else 0,
        "mean_learning_speed": np.mean(fold_learning_speeds) if fold_learning_speeds else 0,
        "mean_convergence_stability": np.mean(fold_convergence_stabilities) if fold_convergence_stabilities else 0,
        "model_path": model_path,
        "effective_hyperparameters": effective_hyperparameters,
        "checkpoint_history": checkpoint_history,  # val metrics every N epochs; empty if checkpoint_every_n_epochs=0
        "inference_time_ms": profiling_results.get("inference_time_ms", 0),
        "params_count": profiling_results.get("params_count", 0),
        "model_size_mb": profiling_results.get("model_size_mb", 0)
    }

    # --- LOG AGGREGATE METRICS TO MLFLOW (Parent Run) ---
    try:
        # We assume the parent run (CV_Experiment) is still active or we can attach to it.
        # Actually, the 'with mlflow.start_run... as parent_run' block closed above?
        # Let's check lines 203... it wraps the loop. This 'final_result' is OUTSIDE the loop.
        # BUT 'training_worker.py' structure has the loop inside 'main'.
        # We need to log these metrics to that same parent run.
        # Since the 'with' block closed, we need to explicitly resume it or move this block inside.
        # BUT we don't have the run_id easily unless we stored it.
        # EASIER FIX: Log these inside the 'finally' of the parent run block, OR just start a new run context here using the same name or ID if available. 
        # Actually, 'mlflow.last_active_run()' might work, but safer to re-open by name? No.
        # Let's check if 'args.experiment_name' is unique enough. Not really.
        # Best approach: Just reopen the run by name? No.
        # Wait, I see 'with mlflow.start_run(...) as parent_run:' at line 203.
        # I should have logged these metrics INSIDE that block.
        # Since I am taking a lazy path, I will put it here but warn it might create a new run if I am not careful.
        # BETTER: I will assume the user mainly cares about the JSON result which IS logged by the AGENT later.
        # BUT the user asked for MLFLOW logging.
        # So I will wrap this logging in a resume block if I can find the ID. 
        # Actually, let's just log them to the *active* run if one exists, or skip.
        # Wait, if the block closed, there is no active run.
        # Re-opening the exact same run requires ID.
        # Let's modify the code to log INSIDE the block in a future step if strictly needed,
        # BUT for now, let's log them to "Final_Model_..." run which IS created below at line 398?
        # No, that's strictly for the final model.
        # Let's add a NEW run "Aggregate_Metrics" or similar? No that's messy.
        
        # CORRECT FIX: The loop at line 203 finishes. The metrics are calculated.
        # I should look at where the 'with' block ends. It ends at line 350 approx.
        # I will replace the whole block calculation to be inside or just log them as a new event.
        # Actually, I will leave it for now in the JSON only but log to the "Final Model" run which is important.
        
        pass 
    except Exception:
        pass
    
    # ----------------------------------------------------
    with open(args.output_path, "w") as f:
        json.dump(final_result, f)
    print(f"Training for {args.experiment_name} on dataset completed successfully.")


import time

def profile_model(model, dataset):
    print("[Profiling] Starting hardware profiling...")
    
    # 1. Inference Time
    # Warmup
    for x, _ in dataset.take(5):
        _ = model(x, training=False)
        
    start_time = time.time()
    num_batches = 0
    # Profile on up to 50 batches
    for x, _ in dataset.take(50): 
        _ = model(x, training=False)
        num_batches += 1
        
    if num_batches == 0:
        return {"inference_time_ms": 0.0, "params_count": 0, "model_size_mb": 0.0}

    end_time = time.time()
    
    avg_inference_time_ms = ((end_time - start_time) / num_batches) * 1000
    print(f"[Profiling] Average Inference Time: {avg_inference_time_ms:.2f} ms/batch")
    
    # 2. Parameter Count (Proxy for memory/complexity)
    total_params = model.count_params()
    
    # 3. Model Size (MB) - Estimate
    # float32 = 4 bytes
    model_size_mb = (total_params * 4) / (1024 * 1024)
    print(f"[Profiling] Model Size: {model_size_mb:.2f} MB")
    
    return {
        "inference_time_ms": avg_inference_time_ms,
        "params_count": total_params,
        "model_size_mb": model_size_mb
    }


def train_supernet(args, strategy, X, y, num_classes, hyperparameters):
    print("[Supernet] Starting Supernet Search...")
    K.clear_session()
    gc.collect()
    
    # Configuration
    epochs = hyperparameters.get('epochs', 50)
    batch_size = hyperparameters.get('batch_size', 64)
    lr = hyperparameters.get('learning_rate', 0.025)
    arch_lr = hyperparameters.get('arch_learning_rate', 3e-4)
    
    # Split data: 50% for weights (train), 50% for architecture (val)
    # This is standard for DARTS to avoid overfitting architecture to training data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
    
    # Create datasets
    # Create datasets
    # Reduce prefetch aggressiveness to avoid Tcache double free issues
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size).prefetch(1)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(batch_size).prefetch(1)
    
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)
    
    print(f"[Supernet Debug] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"[Supernet Debug] Train Dataset element spec: {train_dataset.element_spec}")
    print(f"[Supernet Debug] Distributed Dataset created.")
    
    with strategy.scope():
        model = supernet.build_supernet(
            input_shape=X.shape[1:], 
            num_classes=num_classes, 
            params=hyperparameters
        )
        
        optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        arch_optimizer = keras.optimizers.Adam(learning_rate=arch_lr)
        # FIX: Distributed strategy requires explicit reduction. We use NONE and reduce manually.
        loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        
        train_acc_metric = keras.metrics.CategoricalAccuracy()
        val_acc_metric = keras.metrics.CategoricalAccuracy()

    # Warmup configuration
    warmup_epochs = hyperparameters.get('warmup_epochs', 5)
    print(f"[Supernet] Warmup phase: {warmup_epochs} epochs")

    # --- FIX: Initialize Optimizer Variables Explicitly ---
    # TensorFlow 2.x tf.function forbids creating variables (like optimizer momentum slots)
    # on a non-first call. Since our train_step uses the optimizer conditionally (only when !warmup),
    # the first time it's used might be deep in the loop, crashing execution.
    # We force initialization here by applying zero gradients once.
    # if hasattr(model, 'arch_parameters') and model.arch_parameters():
    #     print("[Supernet] Initializing architecture optimizer variables with dummy step...")
    #     try:
    #         # Create zero gradients matching the architecture parameters
    #         dummy_grads = [tf.zeros_like(p) for p in model.arch_parameters()]
    #         # Apply them once to create the optimizer's internal state variables
    #         arch_optimizer.apply_gradients(zip(dummy_grads, model.arch_parameters()))
    #         print("[Supernet] Optimizer variables initialized successfully.")
    #     except Exception as e:
    #         print(f"[Supernet Warning] Failed to force-init optimizer: {e}")
    # ----------------------------------------------------

    @tf.function
    def train_step(images, labels, val_images, val_labels, warmup):
        # 1. Update Architecture Parameters (Alphas) - SKIPPED IF WARMUP
        arch_loss = 0.0
        if not warmup:
            with tf.GradientTape() as tape:
                val_logits = model(val_images, training=True) 
                per_example_arch_loss = loss_fn(val_labels, val_logits)
                arch_loss = tf.nn.compute_average_loss(per_example_arch_loss, global_batch_size=tf.shape(val_labels)[0])
            
            arch_grads = tape.gradient(arch_loss, model.arch_parameters())
            arch_optimizer.apply_gradients(zip(arch_grads, model.arch_parameters()))
        
        # 2. Update Network Weights
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            per_example_loss = loss_fn(labels, logits)
            loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=tf.shape(labels)[0])
            
        grads = tape.gradient(loss, model.trainable_variables)
        weight_grads = [g for g, v in zip(grads, model.trainable_variables) if 'alphas' not in v.name]
        weight_vars = [v for v in model.trainable_variables if 'alphas' not in v.name]
        
        optimizer.apply_gradients(zip(weight_grads, weight_vars))
        
        train_acc_metric.update_state(labels, logits)
        return loss, arch_loss

    @tf.function
    def val_step(images, labels):
        logits = model(images, training=False)
        val_acc_metric.update_state(labels, logits)

    # Early Stopping Variables
    best_val_acc = 0.0
    patience = 15 # Increased from 5 to allow more exploration
    wait = 0
    best_weights = None

    # Training Loop
    for epoch in range(epochs):
        is_warmup = epoch < warmup_epochs
        status = "WARMUP" if is_warmup else "SEARCH"
        
        # Reduced Verbosity: Log start only every 5 epochs or first/last
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"\nStart of epoch {epoch} [{status}]")
        
        # Iterate over both datasets simultaneously
        train_iter = iter(train_dist_dataset)
        val_iter = iter(val_dist_dataset)
        
        # DEBUG: Print calculated steps (Only once or on error)
        train_len = len(train_dataset)
        val_len = len(val_dataset)
        steps = min(train_len, val_len)
        # print(f"[Supernet Debug] Epoch {epoch}: Train Batches={train_len}, Val Batches={val_len}, Steps={steps}")
        
        for step in range(steps):
            # HEARTBEAT: Print progress every 10% or at least every 100 steps to prove liveness
            # Using flush=True to bypass buffering freezes
            if (steps > 10 and step % (steps // 10) == 0) or step == 0:
                 print(f"[Supernet Progress] Epoch {epoch}: Step {step}/{steps} ({(step/steps)*100:.0f}%)", flush=True)

            try:
                images, labels = next(train_iter)
                val_images, val_labels = next(val_iter)
            except StopIteration:
                print(f"[Supernet Debug] StopIteration encountered at step {step}/{steps}", flush=True)
                break
                
            strategy.run(train_step, args=(images, labels, val_images, val_labels, is_warmup))
            
        train_acc = train_acc_metric.result()
        # Only print accuracy if we printed the start of epoch header
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Training acc over epoch: {float(train_acc):.4f}", flush=True)
        mlflow.log_metric("train_accuracy", float(train_acc), step=epoch)
        # Fix: reset_states -> reset_state
        train_acc_metric.reset_state()
        
        # Validation
        for x_batch_val, y_batch_val in val_dist_dataset:
            strategy.run(val_step, args=(x_batch_val, y_batch_val))
            
        val_acc = val_acc_metric.result()
        print(f"Validation acc: {float(val_acc):.4f}")
        mlflow.log_metric("val_accuracy", float(val_acc), step=epoch)
        # Fix: reset_states -> reset_state
        val_acc_metric.reset_state()
        
        # Early Stopping Logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            best_weights = model.get_weights()
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                model.set_weights(best_weights)
                break
        
    # Extract Genotype
    genotype = model.genotype()
    print("Genotype extracted:", genotype)
    
    # --- VISUALIZATION ---
    try:
        from graphviz import Digraph
        
        def plot_cell(genes, filename):
            g = Digraph(
                format='png',
                edge_attr=dict(fontsize='20', fontname="times"),
                node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
                engine='dot'
            )
            g.body.extend(['rankdir=LR'])
            
            # Input nodes
            g.node("c_{k-2}", fillcolor='darkseagreen2')
            g.node("c_{k-1}", fillcolor='darkseagreen2')
            
            steps = len(genes) // 2
            for i in range(steps):
                g.node(str(i), fillcolor='lightblue')
                
            for i in range(steps):
                for (op, j) in genes[2*i:2*i+2]:  # 2 edges per step
                    # Determine source node label based on input index
                    # j=0 -> c_{k-2}, j=1 -> c_{k-1}, j>=2 -> intermediate node
                    if j == 0: 
                        src_label = "c_{k-2}"
                    elif j == 1: 
                        src_label = "c_{k-1}"
                    else: 
                        src_label = str(j-2)
                    
                    target_label = str(i)
                    g.edge(src_label, target_label, label=op, fillcolor="gray")
            
            g.node("c_{k}", fillcolor='palegoldenrod')
            for i in range(steps):
                g.edge(str(i), "c_{k}", fillcolor="gray")
                
            g.render(filename)
            return f"{filename}.png"

        # Plot Normal Cell
        plot_cell(genotype['normal'], os.path.join(PERSISTENT_PATHS["models_dir"], "normal_cell"))
        plot_cell(genotype['reduce'], os.path.join(PERSISTENT_PATHS["models_dir"], "reduce_cell"))
        
        mlflow.log_artifact(os.path.join(PERSISTENT_PATHS["models_dir"], "normal_cell.png"), "architecture_viz")
        mlflow.log_artifact(os.path.join(PERSISTENT_PATHS["models_dir"], "reduce_cell.png"), "architecture_viz")
        print("[Supernet] Visualization generated and logged.")

    except ImportError:
        print("[Supernet Warning] Graphviz not installed. Skipping visualization.")
    except Exception as e:
        print(f"[Supernet Warning] Visualization failed: {e}")
    # ---------------------

    # Save results
    # Save results
    # Calculate advanced metrics for Supernet
    train_acc_history = [] # We need to capture history. Currently strictly logging to MLFlow.
    # To fix this properly without changing the whole loop structure, we can just grab the FINAL values 
    # since we don't have the full history list easily accessible here (it was printed/logged but not stored in a list).
    # actually, let's just use the final epoch's values as a proxy or 0 if not available.
    
    # Better approach: We can't retroactively calculate std-dev without the list.
    # But for now, let's just return what we have and maybe add simple "final_overfitting"
    
    # Wait, I can't easily add history tracking without modifying the loop above.
    # Let's Modify the loop above to store history? No, too invasive for this step.
    # Let's just calculate it from the LAST epoch values we have variables for?
    # We have train_acc and val_acc from the last loop iteration.
    
    final_train_acc = float(train_acc)
    final_val_acc = float(val_acc)
    overfitting_score = final_train_acc - final_val_acc
    
    result = {
        "status": "completed",
        "genotype": genotype,
        "accuracy": float(val_acc),
        "epochs": epochs,
        "overfitting_score": overfitting_score,
        "train_accuracy": final_train_acc
    }
    
    with open(args.output_path, "w") as f:
        json.dump(result, f)
        
    # Save supernet weights
    if hyperparameters.get("save_model", False):
        model_dir = PERSISTENT_PATHS["models_dir"]
        os.makedirs(model_dir, exist_ok=True)
        model_filename = f"supernet_{os.path.basename(args.output_path).replace('.json', '')}.weights.h5"
        model_path = os.path.join(model_dir, model_filename)
        model.save_weights(model_path)
        print(f"Supernet weights saved to {model_path}")
    else:
        print("[Supernet] 'save_model' is False or missing. Weights NOT saved.")
    
    # Cleanup
    del model
    K.clear_session()
    gc.collect()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker process for model training.")
    parser.add_argument("--experiment_name", required=False, help="Required for 'train' mode")
    parser.add_argument("--params_json", required=False, help="Required for 'train' mode")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--custom_architecture_file", default=None)
    parser.add_argument("--data_path_root", required=False, help="Root directory of the dataset. Required for 'train' mode.")
    parser.add_argument("--test_data_folder", default=None, help="Optional external test set folder")
    parser.add_argument("--mode", default="train", choices=["train", "supernet"], help="Execution mode: 'train' or 'supernet'")
    
    # Supernet specific args
    parser.add_argument("--dataset_id", required=False, help="Required for 'supernet' mode")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for supernet training")
    parser.add_argument("--gpu_id", type=int, default=None, help="Specific GPU ID to use")
    
    parsed_args = parser.parse_args()
    
    # Validate args based on mode
    if parsed_args.mode == "train":
        if not all([parsed_args.experiment_name, parsed_args.params_json, parsed_args.data_path_root]):
            parser.error("--experiment_name, --params_json, and --data_path_root are required for 'train' mode.")
    elif parsed_args.mode == "supernet":
        if not all([parsed_args.dataset_id, parsed_args.data_path_root]):
            parser.error("--dataset_id and --data_path_root are required for 'supernet' mode.")
            
    main(parsed_args)