import os
import re
import mlflow
import traceback
from functools import wraps
from typing import Optional, Dict, Any

# Configure MLFlow Tracking URI
def _get_dynamic_mlflow_uri():
    try:
        from state_manager import PROJECT_ROOT, PERSISTENT_PATHS
        
        # Use the defined mlruns_dir from state_manager (absolute path in processed_data)
        _mlruns_path = PERSISTENT_PATHS.get("mlruns_dir", os.path.join(PROJECT_ROOT, "mlruns"))
        
        # DEBUG: Print paths for diagnosis
        # print(f"[MLFlow Path Debug] PROJECT_ROOT: {PROJECT_ROOT}")
        # print(f"[MLFlow Path Debug] _mlruns_path: {_mlruns_path}")
        
        # Robust Path Normalization for URI
        # On Windows, we need file:///C:/path/to/mlruns
        # On Linux, we need file:///app/path/to/mlruns
        
        # 1. Get absolute path
        abs_mlruns = os.path.abspath(_mlruns_path)
        clean_path = abs_mlruns.replace("\\", "/")
        
        # --- LINUX PATH FIX ---
        # If we are on Linux but the path contains a Windows drive letter (e.g., from a shared config or state)
        if os.name != 'nt' and re.match(r"^[A-Za-z]:", clean_path.lstrip("/")):
            # Strip the drive letter and keep the rest as an absolute Linux path
            clean_path = re.sub(r"^/?([A-Za-z]:)?", "", clean_path)
            if not clean_path.startswith("/"):
                clean_path = f"/{clean_path}"
            # Update abs_mlruns to match the corrected path for local file operations
            abs_mlruns = clean_path
            # print(f"[MLFlow Path Debug] Corrected Windows path for Linux: {abs_mlruns}")
        # ----------------------
        
        if ":" in clean_path and not clean_path.startswith("/") and os.name == 'nt':
             # It's a Windows absolute path (e.g. D:/...) on a Windows host
             uri = f"file:///{clean_path}"
        else:
             # It's a Linux absolute path (e.g. /app/...)
             # Ensure it starts with a single slash if it doesn't already
             if not clean_path.startswith("/"):
                 clean_path = f"/{clean_path}"
             uri = f"file://{clean_path}"
        
        # Final cleanup for double/triple slashes
        uri = uri.replace("file:////", "file:///")
        return uri, abs_mlruns

    except ImportError:
        abs_fallback = os.path.abspath("./mlruns")
        clean_fallback = abs_fallback.replace("\\", "/")
        return f"file://{clean_fallback}", abs_fallback

_calculated_uri, _mlruns_path = _get_dynamic_mlflow_uri()

# --- DOCKER PATH CORRECTION ---
# If we are in Docker (os.path.exists('/app')) and our calculated path 
# includes a long prefix that doesn't exist, but /app/processed_data does,
# we should favor /app/processed_data for persistence.
if not os.name == 'nt' and os.path.exists('/app/processed_data'):
    if "/processed_data/mlruns" in _mlruns_path:
        # Re-calculate to use the standard mount point
        _mlruns_path = "/app/processed_data/mlruns"
        _calculated_uri = "file:///app/processed_data/mlruns"
        # print(f"[MLFlow Docker] Detected Docker mount, re-mapped tracking to: {_calculated_uri}")
# ------------------------------

# Ensure directory exists
if not os.path.exists(_mlruns_path):
    try:
        os.makedirs(_mlruns_path, exist_ok=True)
        # print(f"[MLFlow Check] Created missing mlruns directory: {_mlruns_path}")
    except Exception as e:
        print(f"[MLFlow Check] Failed to create {_mlruns_path}: {e}")

# CRITICAL FIX: Respect env var ONLY if it's valid and exists, otherwise use calculated
_env_uri = os.environ.get("MLFLOW_TRACKING_URI")
if _env_uri:
    # If it's a relative path in config or env, resolve it to absolute
    if _env_uri.startswith("file://./"):
        rel_part = _env_uri.replace("file://./", "")
        abs_path = os.path.abspath(rel_part).replace("\\", "/")
        MLFLOW_TRACKING_URI = f"file:///{abs_path}" if ":" in abs_path else f"file://{abs_path}"
    # Validation: If we are on Windows but the env var looks like Linux (/app/), ignore it.
    elif os.name == 'nt' and ("/app/" in _env_uri or _env_uri.startswith("file:///app")):
        # print(f"[MLFlow] Ignored invalid Linux-style tracking URI on Windows: {_env_uri}")
        MLFLOW_TRACKING_URI = _calculated_uri
    else:
        MLFLOW_TRACKING_URI = _env_uri
else:
    MLFLOW_TRACKING_URI = _calculated_uri

# Final URI cleanup
MLFLOW_TRACKING_URI = MLFLOW_TRACKING_URI.replace("file:////", "file:///")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# print(f"[MLFlow] Tracking URI set to: {MLFLOW_TRACKING_URI}")

def _repair_mlflow_meta_paths(local_mlruns_path: str):
    """
    Dynamically self-heals MLflow meta.yaml files when migrating between 
    Windows (e.g., D:/...) and Linux (e.g., /mnt/...) shared filesystems.
    It rewrites artifact_location and artifact_uri to match the current OS absolute path.
    """
    import os, re
    if not os.path.exists(local_mlruns_path):
        return

    clean_base = os.path.abspath(local_mlruns_path).replace("\\", "/")
    if not clean_base.startswith("/"):
        clean_base = f"/{clean_base}"
    
    target_prefix = f"file://{clean_base}"
    
    # Regex finds file:/// followed by anything up to /mlruns (case insensitive)
    # Examples: file:///D:/.../mlruns or file:///mnt/.../mlruns
    pattern = re.compile(r"file:///(?:[A-Za-z]:/)?(?:[^ \n<>'\"{}]+?)(?i:/mlruns)")
    
    try:
        for root, dirs, files in os.walk(local_mlruns_path):
            if "meta.yaml" in files:
                meta_path = os.path.join(root, "meta.yaml")
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    def replacer(match):
                        return target_prefix
                    
                    new_content, count = pattern.subn(replacer, content)
                    
                    if count > 0 and new_content != content:
                        with open(meta_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                except Exception:
                    pass
    except Exception as e:
        print(f"[MLFlow Repair] Warning during path repair: {e}")

def bootstrap_mlflow(tracking_uri: Optional[str] = None):
    """
    Ensures Experiment 0 exists and sets MLFLOW_TRACKING_URI in the environment
    so subprocesses inherit it.
    """
    global MLFLOW_TRACKING_URI
    
    target_uri = tracking_uri if tracking_uri else MLFLOW_TRACKING_URI
    
    # --- LINUX PATH FIX FOR EXPLICITLY PASSED URIs ---
    # When agent.py (running on Windows) passes its tracking_uri to Celery (running on Linux),
    # the URI might look like file:///D:/... which Linux interprets as a root directory /D:
    if os.name != 'nt' and target_uri.startswith("file:///"):
        # Extract the path part
        path_part = target_uri[8:] # after file:///
        if re.match(r"^[A-Za-z]:", path_part):
            # Strip the drive letter
            clean_path_part = re.sub(r"^[A-Za-z]:", "", path_part)
            if not clean_path_part.startswith("/"):
                clean_path_part = f"/{clean_path_part}"
            target_uri = f"file://{clean_path_part}"
            # print(f"[MLFlow Bootstrap] Re-wrote passed Windows URI for Linux: {target_uri}")
    # -------------------------------------------------
    
    # Environment Hijacking
    os.environ["MLFLOW_TRACKING_URI"] = target_uri
    mlflow.set_tracking_uri(target_uri)
    
    # Ensure Experiment 0 and Run active path repairs
    try:
        # Extract the local path from the URI
        if target_uri.startswith("file://"):
            local_path = target_uri.replace("file:///", "") if target_uri.startswith("file:///") else target_uri.replace("file://", "")
            
            # Handle Windows paths like C:/path or /C:/path
            if local_path.startswith("/") and len(local_path) > 2 and local_path[2] == ':':
                local_path = local_path[1:]
            
            local_path = os.path.abspath(local_path)
            
            # --- AUTO-REPAIR EXISTING METADATA ---
            _repair_mlflow_meta_paths(local_path)
            # -------------------------------------
            exp_0_dir = os.path.join(local_path, "0")
            meta_yaml_path = os.path.join(exp_0_dir, "meta.yaml")
            
            if not os.path.exists(meta_yaml_path):
                os.makedirs(exp_0_dir, exist_ok=True)
                clean_local = local_path.replace("\\", "/")
                with open(meta_yaml_path, "w", encoding="utf-8") as f:
                    f.write(f"artifact_location: file:///{clean_local}/0\n")
                    f.write("creation_time: 1700000000000\n")
                    f.write("experiment_id: '0'\n")
                    f.write("last_update_time: 1700000000000\n")
                    f.write("lifecycle_stage: active\n")
                    f.write("name: Default\n")
    except Exception as e:
        print(f"[MLFlow Bootstrap] Failed to ensure Experiment 0 for {target_uri}: {e}")

# Call it during module initialization
bootstrap_mlflow()


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Ensures the experiment exists and returns its ID.
    Now with better diagnostics for cluster environments and robust fallback.
    """
    try:
        # Normalize name to avoid weird characters in folder names
        experiment_name = str(experiment_name).strip()
        
        # Ensure tracking URI is set before any call
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        else:
            try:
                # Attempt to create
                exp_id = mlflow.create_experiment(experiment_name)
                print(f"[MLFlow] Created new experiment '{experiment_name}' with ID {exp_id}")
                return exp_id
            except Exception as e:
                # Check for "already exists" - sometimes get_by_name fails but create fails too
                err_str = str(e).lower()
                if "already exists" in err_str:
                     experiment = mlflow.get_experiment_by_name(experiment_name)
                     if experiment: return experiment.experiment_id

                print(f"[MLFlow] CRITICAL: Failed to create experiment '{experiment_name}'. Error: {e}")
                
                # FALLBACK LOGIC: Try to find ANY existing experiment
                try:
                    all_exps = mlflow.search_experiments()
                    if all_exps:
                        fallback_exp = all_exps[0]
                        print(f"[MLFlow] FALLBACK: Using existing experiment '{fallback_exp.name}' (ID: {fallback_exp.experiment_id})")
                        return fallback_exp.experiment_id
                except: pass

                # ULTIMATE FALLBACK: Experiment 0
                # Try to ensure it's loaded/created
                try:
                    default_exp = mlflow.get_experiment("0")
                    if default_exp: return "0"
                    # If not found, create 'Default' which is often ID 0
                    return mlflow.create_experiment("Default")
                except:
                    return "0"
    except Exception as e:
        print(f"[MLFlow] Error accessing tracking server: {e}")
        return "0"

# --- ROBUST ARTIFACT LOGGING (FIX FOR /app PERMISSION DENIED) ---
def log_agent_artifact(run_id: str, artifact_name: str, content: str, artifact_path: str = "thoughts"):
    """
    Logs a text content as an artifact to a specific Run ID.
    Handles 'permission denied' errors on cluster by saving to local temporary file 
    instead of crashing the agent.
    """
    try:
        # Pre-emptive cleanup of names
        artifact_name = str(artifact_name).replace(" ", "_")
        
        # Ensure tracking URI is set
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Helper to perform the actual logging
        def _perform_log():
            try:
                mlflow.log_text(content, f"{artifact_path}/{artifact_name}")
            except (OSError, Exception) as e:
                # Catch PermissionError (Errno 13) or other MLFlow errors
                err_str = str(e).lower()
                if "permission denied" in err_str or getattr(e, 'errno', 0) == 13:
                    # TRUE FALLBACK: Write to local disk, not MLFlow
                    try:
                        # Use a persistent path for backup that is likely writable
                        # Fallback to temp directory if persistent path fails permissions too
                        import tempfile
                        backup_base = tempfile.gettempdir()
                        fallback_dir = os.path.join(backup_base, "mlflow_backups", run_id, artifact_path)
                        
                        os.makedirs(fallback_dir, exist_ok=True)
                        fallback_file = os.path.join(fallback_dir, artifact_name)
                        with open(fallback_file, "w", encoding="utf-8") as f:
                            f.write(content)
                        # print(f"[MLFlow] Saved backup artifact (Permission Denied): {fallback_file}")
                    except Exception as e2:
                        print(f"[MLFlow] Backup save failed for {artifact_name}: {e2}")
                else:
                    print(f"[MLFlow] Non-permission error during log_text for {artifact_name}: {e}")

        active_run = mlflow.active_run()
        if active_run and active_run.info.run_id == run_id:
            _perform_log()
        else:
            # resume the run. 
            # CRITICAL FIX: Do not attempt to resume if run_id is clearly offline or not found
            if run_id.startswith("offline_run_"):
                # Still try _perform_log as it might handle lack of active run via fallback
                _perform_log()
                return

            try:
                with mlflow.start_run(run_id=run_id, nested=True):
                     _perform_log()
            except Exception as e_start:
                # If the run doesn't exist, we fallback to log text locally if _perform_log supports it
                # or just ignore to prevent crash.
                err_str = str(e_start).lower()
                if "not found" in err_str or "unauthorized" in err_str:
                    log(f"[MLFlow] Skipping artifact log for {artifact_name}: Run {run_id} not found.")
                    _perform_log() # Attempt fallback inside _perform_log
                else:
                    print(f"[MLFlow] Failed to start run {run_id} for artifact {artifact_name}: {e_start}")
                    # Try one more time without nested if it was a nesting error
                    if "nested" in str(e_start).lower():
                        try:
                            with mlflow.start_run(run_id=run_id):
                                _perform_log()
                        except: pass

    except Exception as e:
        print(f"[MLFlow] Error in log_agent_artifact for {artifact_name}: {e}")

def log_agent_metrics(run_id: str, metrics: Dict[str, float], step: Optional[int] = None):
    """
    Logs metrics to a specific Run ID.
    """
    try:
        active_run = mlflow.active_run()
        if active_run and active_run.info.run_id == run_id:
             mlflow.log_metrics(metrics, step=step)
        elif not run_id.startswith("offline_run_"):
            try:
                with mlflow.start_run(run_id=run_id, nested=True):
                    mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                log(f"[MLFlow] Skipping metrics for run {run_id}: {e}")
    except Exception as e:
        print(f"[MLFlow] Error logging metrics to run {run_id}: {e}")

class MLFlowAgentContext:
    """
    Context Manager to handle MLFlow run lifecycles for Agents.
    Usage:
        with MLFlowAgentContext(experiment="Agent_Brain", run_name="Turn_1") as run:
            ...
    """
    def __init__(self, experiment_name: str, run_name: Optional[str] = None, run_id: Optional[str] = None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run_id = run_id
        # Set URI before getting experiment
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.experiment_id = get_or_create_experiment(experiment_name)
        self.active_run = None

    def __enter__(self):
        try:
            # Ensure URI is set in current thread/context
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            
            if self.run_id:
                # Resume existing run
                self.active_run = mlflow.start_run(run_id=self.run_id)
            else:
                # Start new run
                # If experiment_id is None, it will default to 0
                exp_id = self.experiment_id if self.experiment_id else "0"
                self.active_run = mlflow.start_run(experiment_id=exp_id, run_name=self.run_name)
            return self.active_run
        except Exception as e:
            print(f"[MLFlow] CRITICAL: Failed to start run (Exp: {self.experiment_name}, ID: {self.experiment_id}): {e}")
            # Return a mock run-like object to prevent crashes if possible, 
            # or just return None and let caller handle
            return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.active_run:
                mlflow.end_run()
        except Exception as e:
            print(f"[MLFlow] Error ending run: {e}")
