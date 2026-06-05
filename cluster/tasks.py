import os
import sys
import json
import subprocess
import traceback
from .celery_app import app

# Ensure project root is in path for imports if needed
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from state_manager import PERSISTENT_PATHS
from system_utils import get_gpu_memory_info, acquire_gpu_lock, release_gpu_lock, is_gpu_locked, logical_to_physical_gpu
from ui_logger import log
from .mlflow_utils import bootstrap_mlflow

# Ensure MLFlow tracking URI is propagated to subprocesses via os.environ
bootstrap_mlflow()

@app.task
def monitor_resources_task():
    """
    Periodic task to check system health (GPU memory, Disk space).
    Logs warnings if resources are critically low.
    """
    try:
        # 1. Check GPU
        gpu_info = get_gpu_memory_info()
        for idx, gpu in enumerate(gpu_info):
            used_mb = gpu['used']
            total_mb = gpu['total']
            utilization = used_mb / total_mb if total_mb > 0 else 0
            if utilization > 0.90:
                log(f"[System Monitor] WARNING: GPU {idx} Memory Critical: {used_mb}/{total_mb} MiB ({utilization:.1%})")
        
        # 2. Check Disk (processed_data)
        # shutil might need import, or use os.statvfs (unix) / ctypes (win)
        # Keeping it simple: Just GPU for now for stability.
        
        return {"status": "success", "gpu_info": gpu_info}
    except Exception as e:
        log(f"[System Monitor] Error: {e}")
        return {"status": "error", "message": str(e)}

def _acquire_free_gpu():
    """Auto-discovers and locks a free GPU for the worker.

    Uses physical_id embedded in get_gpu_memory_info() entries so that
    lock files and CUDA_VISIBLE_DEVICES always reference physical GPU IDs,
    regardless of whether CUDA_VISIBLE_DEVICES was set in the environment
    before this call (e.g. auto-isolation on DGX nodes).
    """
    try:
        gpu_info = get_gpu_memory_info()
        if not gpu_info:
            return None
        # Sort by free memory descending, prefer least-used GPU
        sorted_gpus = sorted(gpu_info, key=lambda x: x['free'], reverse=True)
        for gpu in sorted_gpus:
            if gpu['free'] > 4000:
                phys_id = gpu.get("physical_id", sorted_gpus.index(gpu))
                if not is_gpu_locked(phys_id):
                    if acquire_gpu_lock(phys_id, timeout=2):
                        log(f"[GPU Manager] Auto-acquired lock for physical GPU {phys_id} (Free: {gpu['free']} MiB)")
                        return phys_id
        log("[GPU Manager] WARNING: No free GPUs with >4GB memory could be locked!")
    except Exception as e:
        log(f"[GPU Manager] Error auto-acquiring GPU: {e}")
    return None

def _run_subprocess_task(args, output_path, env=None):
    """Helper to run subprocess and handle output. Accepts an optional custom env."""
    try:
        if env is None:
            # ENVIRONMENT FIX: Prevent OOM on DGX by forcing memory growth
            env = os.environ.copy()
            env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

        # --- GLOBAL FRAMEWORK STABILITY ---
        env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
        # ----------------------------------
        
        # --- DETERMINISM & XLA FIX ---
        # 3D operations on A100 often lack deterministic implementations.
        # Enforce non-deterministic mode to allow training to proceed.
        env["TF_DETERMINISTIC_OPS"] = "0"
        env["TF_CUDNN_DETERMINISTIC"] = "0"
        
        # Explicitly disable XLA deterministic ops to resolve UnimplementedError
        current_xla = env.get("XLA_FLAGS", "")
        if "--xla_gpu_deterministic_ops" not in current_xla:
             env["XLA_FLAGS"] = f"{current_xla} --xla_gpu_deterministic_ops=false".strip()
        # -----------------------------

        # Write subprocess stdout/stderr to a timestamped log file for immediate visibility.
        # This replaces capture_output=True so failures are visible without waiting for completion.
        import time as _time
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        script_name = os.path.basename(args[1]) if len(args) > 1 else "subprocess"
        cuda_dev = env.get("CUDA_VISIBLE_DEVICES", "cpu")
        ts = int(_time.time())
        log_path = os.path.join(log_dir, f"worker_{script_name}_gpu{cuda_dev}_{ts}.log")

        with open(log_path, "w") as log_fh:
            log(f"[SubprocessLog] Writing output to {log_path}")
            result = subprocess.run(args, stdout=log_fh, stderr=subprocess.STDOUT, text=True, check=False, env=env)

        # Read back for return value (keep existing callers working)
        with open(log_path, "r") as log_fh:
            captured_output = log_fh.read()

        if result.returncode != 0:
            return {
                "status": "error",
                "message": f"Process failed with code {result.returncode}",
                "stderr": captured_output,
                "stdout": captured_output,
                "log_path": log_path,
            }
            
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as decode_error:
                return {
                    "status": "error",
                    "message": f"Process finished but output file was not valid JSON. Decode error: {decode_error}",
                    "stderr": captured_output,
                    "stdout": captured_output,
                    "log_path": log_path,
                }
        else:
            return {
                "status": "error",
                "message": "Process finished but output file not found.",
                "stderr": captured_output,
                "stdout": captured_output,
                "log_path": log_path,
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Task exception: {str(e)}",
            "traceback": traceback.format_exc()
        }

@app.task(bind=True, time_limit=172800)  # 48h limit
def run_training_worker(self, experiment_name, params, output_path, data_path_root=None, custom_architecture_file=None, test_data_folder=None, gpu_id=None):
    if not output_path:
        return {"status": "error", "message": "output_path provided to run_training_worker is None"}
        
    script_path = os.path.join(PERSISTENT_PATHS["scripts_dir"], "training_worker.py")
    
    # Ensure all arguments are strings and not None
    args = [
        "python", str(script_path),
        "--experiment_name", str(experiment_name if experiment_name else "default_experiment"),
        "--params_json", json.dumps(params) if params is not None else "{}",
        "--output_path", str(output_path)
    ]
    # Removed conditional add of data_path_root here as we handle it centrally below
    # if data_path_root:
    #     args.extend(["--data_path_root", str(data_path_root)])
    if custom_architecture_file:
        args.extend(["--custom_architecture_file", str(custom_architecture_file)])
    if test_data_folder:
        args.extend(["--test_data_folder", str(test_data_folder)])
    
    # --- GPU ISOLATION FIX ---
    auto_locked_gpu = None
    if gpu_id is None:
        auto_locked_gpu = _acquire_free_gpu()
        if auto_locked_gpu is not None:
             gpu_id = auto_locked_gpu

    if gpu_id is not None:
         args.extend(["--gpu_id", str(gpu_id)])
    # -------------------------

    # --- PATH HARDENING FIX ---
    # Ensure data_path_root is absolute. If None, default to generated_datasets_dir
    final_data_root = data_path_root if data_path_root else PERSISTENT_PATHS["generated_datasets_dir"]
    args.extend(["--data_path_root", str(final_data_root)])
    # --------------------------

    try:
        # ENVIRONMENT ISOLATION: Resolve physical GPU ID and inject into env BEFORE launch.
        # This ensures TensorFlow sees only the target GPU even before main() runs.
        worker_env = os.environ.copy()
        worker_env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        if gpu_id is not None:
            # gpu_id is already a physical ID (returned by _acquire_free_gpu via physical_id field).
            # Do NOT call logical_to_physical_gpu here — it would re-translate using os.environ
            # which may not reflect the auto-isolation computed by get_gpu_memory_info().
            worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            log(f"[Task Manager] Launching training on Physical GPU {gpu_id}")
        
        return _run_subprocess_task(args, output_path, env=worker_env)
    finally:
        if auto_locked_gpu is not None:
            release_gpu_lock(auto_locked_gpu)

@app.task(bind=True, time_limit=172800)  # 48h limit
def run_finetune_training_worker(self, experiment_name, params, output_path,
                                  data_path_root=None, custom_architecture_file=None,
                                  seeds=None, gpu_id=None):
    """
    Two-phase pretrain + finetune worker for committed architectures.
    Dispatches to processed_data/scripts_worker/finetune_training_worker.py.
    """
    if not output_path:
        return {"status": "error", "message": "output_path is None"}

    script_path = os.path.join(PERSISTENT_PATHS["scripts_dir"], "finetune_training_worker.py")

    args = [
        "python", str(script_path),
        "--experiment_name", str(experiment_name or "default_experiment"),
        "--params_json", json.dumps(params) if params is not None else "{}",
        "--output_path", str(output_path),
    ]
    if custom_architecture_file:
        args.extend(["--custom_architecture_file", str(custom_architecture_file)])
    if seeds:
        args.extend(["--seeds_json", json.dumps(seeds)])

    # --- GPU ISOLATION FIX (mirrors run_training_worker) ---
    auto_locked_gpu = None
    if gpu_id is None:
        auto_locked_gpu = _acquire_free_gpu()
        if auto_locked_gpu is not None:
            gpu_id = auto_locked_gpu
    if gpu_id is not None:
        args.extend(["--gpu_id", str(gpu_id)])

    # --- PATH HARDENING FIX ---
    final_data_root = data_path_root if data_path_root else PERSISTENT_PATHS["generated_datasets_dir"]
    args.extend(["--data_path_root", str(final_data_root)])

    try:
        worker_env = os.environ.copy()
        worker_env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        if gpu_id is not None:
            worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            log(f"[Task Manager] Launching finetune-training on Physical GPU {gpu_id}")
        return _run_subprocess_task(args, output_path, env=worker_env)
    finally:
        if auto_locked_gpu is not None:
            release_gpu_lock(auto_locked_gpu)


@app.task(bind=True)
def run_supernet_worker(self, dataset_id, params, output_path, data_path_root=None, gpu_id=None):
    script_path = os.path.join(PERSISTENT_PATHS["scripts_dir"], "training_worker.py")
    args = [
        "python", script_path,
        "--mode", "supernet",
        "--dataset_id", dataset_id,
        "--params_json", json.dumps(params),
        "--output_path", output_path
    ]

    # Pass epochs explicitly if present (training_worker expects flag)
    if "epochs" in params:
        args.extend(["--epochs", str(params["epochs"])])

    # --- GPU ISOLATION FIX ---
    auto_locked_gpu = None
    if gpu_id is None:
        auto_locked_gpu = _acquire_free_gpu()
        if auto_locked_gpu is not None:
             gpu_id = auto_locked_gpu

    if gpu_id is not None:
         args.extend(["--gpu_id", str(gpu_id)])
    # -------------------------

    # --- PATH HARDENING FIX ---
    final_data_root = data_path_root if data_path_root else PERSISTENT_PATHS["generated_datasets_dir"]
    args.extend(["--data_path_root", str(final_data_root)])
    # --------------------------

    try:
        # ENVIRONMENT ISOLATION: Resolve physical GPU ID and inject into env BEFORE launch.
        worker_env = os.environ.copy()
        worker_env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        if gpu_id is not None:
            phys_id = logical_to_physical_gpu(gpu_id)
            worker_env["CUDA_VISIBLE_DEVICES"] = str(phys_id)
            log(f"[Task Manager] Launching supernet on Physical GPU {phys_id} (Logical {gpu_id})")
        
        return _run_subprocess_task(args, output_path, env=worker_env)
    finally:
        if auto_locked_gpu is not None:
            release_gpu_lock(auto_locked_gpu)

@app.task(bind=True, time_limit=172800)  # 48h limit
def run_data_pipeline_worker(self, raw_data_source, manifest_name, representation_type, params, output_path):
    script_path = os.path.join(PERSISTENT_PATHS["scripts_dir"], "data_pipeline_worker.py")
    
    # --- PATH HARDENING FIX ---
    # Resolve raw_data_source to absolute path
    if os.path.isabs(raw_data_source) and os.path.exists(raw_data_source):
        final_source = raw_data_source
    else:
        final_source = os.path.join(PERSISTENT_PATHS["raw_data_dir"], raw_data_source)
    # --------------------------

    args = [
        "python", script_path,
        "--raw_data_source", str(final_source),
        "--manifest_name", manifest_name,
        "--representation_type", representation_type,
        "--params_json", json.dumps(params),
        "--output_path", output_path
    ]
    return _run_subprocess_task(args, output_path)

@app.task(bind=True)
def run_ensemble_evaluation_worker(self, dataset_id, output_path):
    script_path = os.path.join(PERSISTENT_PATHS["scripts_dir"], "ensemble_evaluation_worker.py")
    args = [
        "python", script_path,
        "--dataset_id", dataset_id,
        "--output_path", output_path
    ]
    return _run_subprocess_task(args, output_path)

# ============================================================
# IMAGENET COMPETITION TASKS
# These tasks are ONLY dispatched when execution.mode = "imagenet_competition".
# They follow the same subprocess pattern as existing training tasks.
# ============================================================

@app.task(bind=True, name="cluster.tasks.run_imagenet_training_task", time_limit=172800)  # 48h limit
def run_imagenet_training_task(self, train_args: dict):
    """
    Celery task: launches imagenet_training_worker.py as a subprocess.

    Args:
        train_args: Dict with all training parameters (dataset_dir, backbone_depth, etc.)
                    Matches the CLI args accepted by imagenet_training_worker.py.

    Returns:
        dict with status, mlflow_run_id, best_checkpoint_path, and metric summary.
    """
    script_path = os.path.join(
        PERSISTENT_PATHS["scripts_dir"], "imagenet_training_worker.py"
    )
    output_path = os.path.join(
        train_args.get("checkpoint_dir", PERSISTENT_PATHS["generated_datasets_dir"]),
        "training_result.json",
    )

    env = os.environ.copy()
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    # Re-enable XLA for this worker subprocess (main.py disables it globally for stability,
    # but the imagenet worker explicitly needs it for performance).
    env.pop("TF_XLA_FLAGS", None)

    args = [
        "python", str(script_path),
        "--args_json", json.dumps(train_args),
        "--output_path", output_path,
    ]

    log(f"[ImageNet Training Task] Launching worker: {script_path}")
    return _run_subprocess_task(args, output_path, env=env)


@app.task(bind=True, name="cluster.tasks.run_imagenet_inference_task", time_limit=21600)  # 6h limit
def run_imagenet_inference_task(self, infer_args: dict):
    """
    Celery task: launches imagenet_inference_worker.py as a subprocess.

    Args:
        infer_args: Dict with checkpoint_path, test_dir, synset_map, output_path, batch_size.

    Returns:
        dict with status, submission_csv_path, num_images_processed.
    """
    script_path = os.path.join(
        PERSISTENT_PATHS["scripts_dir"], "imagenet_inference_worker.py"
    )
    output_path = os.path.join(
        infer_args.get("output_path", PERSISTENT_PATHS["generated_datasets_dir"]),
        "inference_result.json",
    )

    env = os.environ.copy()
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    args = [
        "python", str(script_path),
        "--args_json", json.dumps(infer_args),
        "--output_path", output_path,
    ]

    log(f"[ImageNet Inference Task] Launching worker: {script_path}")
    return _run_subprocess_task(args, output_path, env=env)

@app.task(bind=True, name="cluster.tasks.run_imagenet_optimization_task", time_limit=172800) # 48h limit
def run_imagenet_optimization_task(self, opt_args: dict):
    """
    Celery task: launches imagenet_opt_worker.py as a subprocess.
    Used by the AI Agent to design and trial architectures on ImageNet.
    """
    script_path = os.path.join(
        PERSISTENT_PATHS["scripts_dir"], "imagenet_opt_worker.py"
    )
    
    # Generate a unique output path for this specific trial
    import uuid
    output_path = os.path.join(
        PERSISTENT_PATHS["generated_datasets_dir"],
        f"imagenet_opt_result_{uuid.uuid4().hex[:8]}.json",
    )

    env = os.environ.copy()
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    env.pop("TF_XLA_FLAGS", None)

    args = [
        "python", str(script_path),
        "--args_json", json.dumps(opt_args),
        "--output_path", output_path,
    ]

    log(f"[ImageNet Optimization Task] Launching worker: {script_path}")
    return _run_subprocess_task(args, output_path, env=env)
