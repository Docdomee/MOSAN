import subprocess
import xml.etree.ElementTree as ET
import os

# --- FRAMEWORK STABILITY FIX ---
# 1. Force Protobuf to use pure-python implementation to avoid 'MessageFactory' attribute errors.
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# 2. Disable oneDNN optimizations to prevent potential registration crashes.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# -------------------------------

def get_gpu_memory_info():
    """
    Retrieves GPU memory information using nvidia-smi.
    Respects CUDA_VISIBLE_DEVICES to map logical IDs to physical stats.

    Returns:
        A list of dictionaries, where each dictionary represents a LOGICAL GPU
        and contains 'total', 'used', and 'free' memory in MiB.
        Returns an empty list if nvidia-smi is not found or fails.
    """
    try:
        result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, check=True)
        root = ET.fromstring(result.stdout)
        
        # 1. Gather all physical GPU stats
        all_gpu_info = []
        for gpu in root.findall("gpu"):
            mem_info = gpu.find("fb_memory_usage")
            total = int(mem_info.find("total").text.replace(" MiB", ""))
            used = int(mem_info.find("used").text.replace(" MiB", ""))
            free = int(mem_info.find("free").text.replace(" MiB", ""))
            all_gpu_info.append({"total": total, "used": used, "free": free})
            
        # 2. Check for environment restrictions (Logical Mapping)
        # We prioritize CUDA_VISIBLE_DEVICES, but fall back to AGENT_GPUS 
        # (the project's own reservation var) if CVD is unset.
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("AGENT_GPUS")
        
        # --- DGX CLUSTER AUTO-ISOLATION ---
        # If no explicit isolation is set and we are on an 8-GPU node (DGX-style),
        # we defensively assume GPUs 0-3 are for vLLM and 4-7 are for training.
        if cvd is None and len(all_gpu_info) == 8:
            vllm_occupied = any(all_gpu_info[i]['used'] > 5000 for i in range(4))
            if vllm_occupied:
                cvd = "4,5,6,7"
                # print("[System] Auto-isolated training to GPUs 4-7 (GPUs 0-3 reserved for vLLM).")
        # ----------------------------------
        
        if cvd is not None:
            logical_gpu_info = []
            for gpu_id_str in cvd.split(','):
                gpu_id_str = gpu_id_str.strip()
                if not gpu_id_str:
                    continue
                try:
                    phys_id = int(gpu_id_str)
                    if 0 <= phys_id < len(all_gpu_info):
                        entry = dict(all_gpu_info[phys_id])
                        entry["physical_id"] = phys_id
                        logical_gpu_info.append(entry)
                    else:
                        logical_gpu_info.append({"total": 0, "used": 0, "free": 0, "physical_id": phys_id})
                except ValueError:
                    pass
            return logical_gpu_info

        # 3. No restrictions: physical_id == logical_id
        for i, info in enumerate(all_gpu_info):
            info["physical_id"] = i
        return all_gpu_info

    except (subprocess.CalledProcessError, FileNotFoundError, ET.ParseError):
        return []


def logical_to_physical_gpu(logical_id: int) -> int:
    """
    Converts a logical GPU index to the physical GPU index.
    Prioritizes CUDA_VISIBLE_DEVICES, falls back to AGENT_GPUS.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("AGENT_GPUS")
    if cvd is None:
        return logical_id
    physical_ids = [int(x.strip()) for x in cvd.split(",") if x.strip().isdigit()]
    if 0 <= logical_id < len(physical_ids):
        return physical_ids[logical_id]
    return logical_id

def setup_gpu_memory():
    """
    Configures TensorFlow to use memory growth (lazy allocation).
    MUST be called at the very start of entry scripts (agent.py, celery_app.py, workers).
    Moved from tools.py to avoid circular imports.
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            return # No GPUs, nothing to do
            
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(f"[System] GPU Setup Warning: {e}")
        
        print(f"[System] GPU Memory Growth Enabled for {len(gpus)} device(s).")
    except ImportError:
        pass # TF not installed, ignore for non-GPU nodes

# --- GPU LOCKING SYSTEM (Concurrency Control) ---
import os
import time
import tempfile
import random

def _get_lock_path(gpu_id):
    # gpu_id must already be a physical GPU ID. Callers obtain it from the
    # physical_id field of get_gpu_memory_info() entries so that auto-isolation
    # (e.g. DGX vLLM reservation) is respected without relying on os.environ.
    # logical_to_physical_gpu is intentionally NOT called here.

    # Use a centralized persistent directory for locks to sync across cluster nodes
    try:
        from state_manager import PERSISTENT_PATHS
        locks_dir = PERSISTENT_PATHS.get("gpu_locks_dir")
    except ImportError:
        locks_dir = None

    if not locks_dir:
        locks_dir = os.path.join(os.getcwd(), "processed_data", "locks")

    os.makedirs(locks_dir, exist_ok=True)
    return os.path.join(locks_dir, f"gpu_phys_{gpu_id}.lock")

def acquire_gpu_lock(gpu_id: int, timeout: int = 10, ttl: int = 3600) -> bool:
    """
    Attempts to acquire an exclusive lock file for the specified GPU.
    
    Args:
        gpu_id: The ID of the GPU to lock.
        timeout: How long to wait (in seconds) to acquire the lock.
        ttl: Time-To-Live in seconds. If lock is older than this, it is considered stale and forced.
    
    Returns:
        True if lock acquired, False otherwise.
    """
    lock_file = _get_lock_path(gpu_id)
    start_time = time.time()
    
    while (time.time() - start_time) < timeout:
        try:
            # Atomic creation: fails if file exists
            fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            
            # Write PID and timestamp for debugging/TTL checking
            with os.fdopen(fd, 'w') as f:
                f.write(f"PID: {os.getpid()}\\nTime: {time.time()}")
            
            return True
        except FileExistsError:
            # Lock exists. Check for staleness (TTL).
            try:
                # Check modification time
                if os.path.exists(lock_file):
                    mtime = os.path.getmtime(lock_file)
                    if (time.time() - mtime) > ttl:
                        print(f"[GPU Lock] Found stale lock for GPU {gpu_id} (Age: {time.time() - mtime:.0f}s). Removing...")
                        try:
                            os.remove(lock_file)
                        except OSError:
                            pass # Race condition: someone else removed it or re-acquired it
                        continue # Retry immediately
            except OSError:
                pass # File might have been removed in the split second, just retry
            
            time.sleep(random.uniform(0.5, 1.5)) # Randomized wait to prevent collisions
        except OSError as e:
            print(f"[GPU Lock] OS Error during acquire: {e}")
            return False
            
    print(f"[GPU Lock] Failed to acquire lock for GPU {gpu_id} after {timeout}s.")
    return False

def release_gpu_lock(gpu_id: int) -> None:
    """Releases the lock for the specified GPU."""
    lock_file = _get_lock_path(gpu_id)
    try:
        if os.path.exists(lock_file):
            os.remove(lock_file)
            # print(f"[GPU Lock] Released lock for GPU {gpu_id}.")
    except Exception as e:
        print(f"[GPU Lock] Error releasing lock for GPU {gpu_id}: {e}")

def is_gpu_locked(gpu_id: int, ttl: int = 3600) -> bool:
    """Checks if a GPU is currently locked."""
    lock_file = _get_lock_path(gpu_id)
    if not os.path.exists(lock_file):
        return False
        
    # Check TTL
    try:
        mtime = os.path.getmtime(lock_file)
        if (time.time() - mtime) > ttl:
            return False # Stale lock
        return True
    except OSError:
        return False # Race condition
