# tools.py
import gc
import importlib.util
import io
import json
import os
import random
import re
import subprocess  # Per lanciare processi esterni
import tempfile  # Per creare file temporanei
import traceback
import time
import uuid
from contextlib import redirect_stdout
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
try:
    import optuna
    from optuna.importance import get_param_importances
except ImportError:
    optuna = None
    get_param_importances = None

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# --- CRITICAL FIX: Explicit GPU Memory Setup ---
# tools.py is imported by Agent and Celery workers. 
# We moved the global side-effect (set_memory_growth) into a function to avoid
# unintentional initialization during import, which causes race conditions with vLLM.

from system_utils import setup_gpu_memory

from tensorflow.keras import backend as K

# Scientific Rigor: Statistical significance check
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    stats = None
    SCIPY_AVAILABLE = False

try:
    from state_manager import PERSISTENT_PATHS, archive_conversation_log, get_todos, update_todos,save_state
except ImportError:
    from state_manager import PERSISTENT_PATHS, archive_conversation_log, get_todos, update_todos, save_state
from ui_logger import log
from model_factory import MODEL_BUILDERS

# Celery Tasks
# Import Celery tasks for distributed execution
from cluster.tasks import (
    run_training_worker,
    run_finetune_training_worker,
    run_supernet_worker,
    run_data_pipeline_worker,
    run_ensemble_evaluation_worker,
    run_imagenet_training_task,
    run_imagenet_inference_task,
    run_imagenet_optimization_task,
)
# tools.py - Add after imports

def _get_mode_or_median(series: pd.Series) -> Union[int, float]:
    """
    Helper function to aggregate discrete parameters correctly.
    For integer-like values (e.g., kernel_size, num_filters): returns Mode as int.
    For continuous values (e.g., learning_rate, dropout): returns Mean as float.
    
    Scientific Rigor Upgrade: Prevents nonsensical values like "Kernel Size: 3.67".
    """
    # Check if all values are integer-like (even if stored as float)
    if pd.api.types.is_integer_dtype(series) or (series.dropna() == series.dropna().astype(int)).all():
        # Discrete → Mode (most frequent value)
        mode_result = series.mode()
        if len(mode_result) > 0:
            return int(mode_result.iloc[0])
        else:
            # Fallback to median if mode fails
            return int(series.median())
    else:
        # Continuous → Mean
        return float(series.mean())

def _get_subprocess_env(gpu_id: Optional[int]) -> Dict[str, str]:
    """
    Creates a copy of the current environment and sets CUDA_VISIBLE_DEVICES
    to the specific gpu_id.
    
    CRITICAL FIX: Maps logical GPU ID (from TF) to Physical GPU ID 
    using the parent's CUDA_VISIBLE_DEVICES if present.
    """
    env = os.environ.copy()
    if gpu_id is not None:
        # Resolve Logical -> Physical mapping
        parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        final_gpu_id = str(gpu_id) # Default fallback
        
        if parent_cvd:
            try:
                # Example: parent_cvd="2,3,4,5"
                # gpu_id=0 -> should be "2"
                available_gpus = [g.strip() for g in parent_cvd.split(',') if g.strip()]
                if 0 <= gpu_id < len(available_gpus):
                    final_gpu_id = available_gpus[gpu_id]
                    log(f"[System] Mapped Logical GPU {gpu_id} -> Physical GPU {final_gpu_id}")
                else:
                     log(f"[System] Warning: GPU ID {gpu_id} out of bounds for CUDA_VISIBLE_DEVICES={parent_cvd}. Using raw {gpu_id}.")
            except Exception as e:
                log(f"[System] Warning: Failed to parse CUDA_VISIBLE_DEVICES: {e}")
        
        env["CUDA_VISIBLE_DEVICES"] = final_gpu_id
        # log(f"[System] Setting CUDA_VISIBLE_DEVICES={final_gpu_id} for subprocess.")
    
    # --- CRITICAL FIX ---
    # Removed "--tf_xla_enable_xla_devices=true" -> Causes duplicate factory registration crash
    # Removed "--xla_gpu_strict_conv_algorithm_picker=false" -> Not needed if XLA is off
    
    # Only keep essential TF flags to reduce noise and manage memory
    env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" 
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    return env
AVAILABLE_TOOLS = {
    # Data tools
    # Defensive wrapper: handles two calling conventions.
    # Convention A (correct): {"manifest_name": "...", "params": {...}, "thread_safe_state": ...}
    # Convention B (agent flat): {"manifest_name": "...", "n_fft": 1024, ..., "thread_safe_state": ...}
    # When the agent sends flat kwargs instead of a nested `params` dict, **kwargs captures them
    # and they are merged into params so the underlying function always receives the right signature.
    "create_dataset_manifest": lambda thread_safe_state, manifest_name=None, params=None, **kwargs: create_dataset_manifest(
        manifest_name=manifest_name or kwargs.pop("dataset_name", None),
        params={**(params or {}), **{k: v for k, v in kwargs.items() if k not in ("thread_safe_state", "dataset_name", "architecture")}},
        thread_safe_state=thread_safe_state,
    ),
    "generate_representation": lambda manifest_name, representation_type, thread_safe_state: generate_representation(manifest_name=manifest_name, representation_type=representation_type, thread_safe_state=thread_safe_state),
    "list_available_datasets": lambda thread_safe_state: list_available_datasets(thread_safe_state=thread_safe_state),
    # Training tools
    "run_training_trial": lambda thread_safe_state, **kwargs: run_training_trial(thread_safe_state=thread_safe_state, **kwargs),
    "run_training_trial_with_finetune": lambda thread_safe_state, **kwargs: run_training_trial_with_finetune(thread_safe_state=thread_safe_state, **kwargs),
    "evaluate_finetune_progress": lambda thread_safe_state, last_n=None, **kwargs: evaluate_finetune_progress(thread_safe_state=thread_safe_state, last_n=last_n),
    "run_imagenet_optimization_trial": lambda thread_safe_state, custom_architecture_file, **kwargs: run_imagenet_optimization_trial(thread_safe_state=thread_safe_state, custom_architecture_file=custom_architecture_file, **kwargs),
    "run_strategic_optuna_sweep": lambda thread_safe_state, **kwargs: run_strategic_optuna_sweep(thread_safe_state=thread_safe_state, **kwargs),
    "run_supernet_search": lambda dataset_id, thread_safe_state, max_epochs=50, population_size=50, evolution_generations=20, representation_type=None, gpu_id=None, **kwargs: run_supernet_search(dataset_id=dataset_id, thread_safe_state=thread_safe_state, max_epochs=max_epochs, population_size=population_size, evolution_generations=evolution_generations, representation_type=representation_type, gpu_id=gpu_id, **kwargs),

    # ----------------------------------------------------------
     # Memory tools
    "memorize_finding": lambda finding, experiment_name, thread_safe_state, manifest_name=None, representation_type=None, category=None: memorize_finding(finding=finding, experiment_name=experiment_name, thread_safe_state=thread_safe_state, manifest_name=manifest_name, representation_type=representation_type, category=category),
    "recall_relevant_memories": lambda query, thread_safe_state, k=5: recall_relevant_memories(query=query, thread_safe_state=thread_safe_state, k=k),
    # Architecture tools
    "write_architecture_file": lambda filename, code, thread_safe_state: write_architecture_file(filename=filename, code=code, thread_safe_state=thread_safe_state),
    "construct_model_from_genotype": lambda genotype, representation_type, thread_safe_state: construct_model_from_genotype(genotype=genotype, representation_type=representation_type, thread_safe_state=thread_safe_state),
    "validate_architecture_file": lambda filename, expected_input_type, thread_safe_state, context_reason="Strategic pivot to new architecture": validate_architecture_file(filename=filename, expected_input_type=expected_input_type, context_reason=context_reason),
    "read_file_from_architectures": lambda filename, thread_safe_state: read_file_from_architectures(filename=filename, thread_safe_state=thread_safe_state),
    "append_to_architecture_changelog": lambda py_filename, log_entry, thread_safe_state: append_to_architecture_changelog(py_filename=py_filename, log_entry=log_entry),
    # Analysis tools
    "get_recent_trials": lambda thread_safe_state, n=5, architecture_filter=None: get_recent_trials(thread_safe_state=thread_safe_state, n=n, architecture_filter=architecture_filter),
    "get_sorted_trials": lambda thread_safe_state, sort_by="mean_accuracy", ascending=False, n=10, architecture_filter=None: get_sorted_trials(thread_safe_state=thread_safe_state, sort_by=sort_by, ascending=ascending, n=n, architecture_filter=architecture_filter),
    "get_correlation_matrix": lambda thread_safe_state, architecture_filter=None: get_correlation_matrix(thread_safe_state=thread_safe_state, architecture_filter=architecture_filter),
    "get_parameter_importance": lambda thread_safe_state, architecture_filter=None, **kwargs: get_parameter_importance(thread_safe_state=thread_safe_state, architecture_filter=architecture_filter),
    "analyze_best_vs_worst_trials": lambda thread_safe_state, architecture_filter=None: analyze_best_vs_worst_trials(thread_safe_state=thread_safe_state, architecture_filter=architecture_filter),
    "summarize_findings": lambda thread_safe_state: summarize_findings(thread_safe_state=thread_safe_state),
    "propose_architectural_mutation": lambda thread_safe_state, **kwargs: propose_architectural_mutation(),
    "get_architecture_summary": lambda experiment_name, custom_architecture_file, thread_safe_state: get_architecture_summary(experiment_name=experiment_name, custom_architecture_file=custom_architecture_file, thread_safe_state=thread_safe_state),
    "analyze_hyperparameter_tradeoffs": lambda thread_safe_state, architecture_filter=None: analyze_hyperparameter_tradeoffs(thread_safe_state=thread_safe_state, architecture_filter=architecture_filter),
    "get_explored_ranges": lambda thread_safe_state, architecture_filter=None: get_explored_ranges(thread_safe_state=thread_safe_state, architecture_filter=architecture_filter),
    "analyze_hyperparameter_trend": lambda param_x, param_y, thread_safe_state, architecture_filter=None: analyze_hyperparameter_trend(param_x=param_x, param_y=param_y, thread_safe_state=thread_safe_state, architecture_filter=architecture_filter),
    "run_architecture_comparison": lambda architecture_A, architecture_B, query, execute_tool_func, thread_safe_state: run_architecture_comparison(architecture_A=architecture_A, architecture_B=architecture_B, query=query, execute_tool_func=execute_tool_func, thread_safe_state=thread_safe_state),
    "run_strategic_graph_evaluation": lambda execute_tool_func, thread_safe_state, **kwargs: run_strategic_graph_evaluation(execute_tool_func=execute_tool_func, thread_safe_state=thread_safe_state, **kwargs),
    "get_strategic_path": lambda thread_safe_state, current_state_vector=None, goal_description=None, max_path_length=10: get_strategic_path(current_state_vector=current_state_vector, goal_description=goal_description, max_path_length=max_path_length, thread_safe_state=thread_safe_state),
    "get_local_transitions": lambda thread_safe_state, current_state_vector=None, top_k=5: get_local_transitions(current_state_vector=current_state_vector, top_k=top_k, thread_safe_state=thread_safe_state),
    "get_hybrid_advice": lambda thread_safe_state, strategic_intent=None: get_hybrid_advice(strategic_intent=strategic_intent, thread_safe_state=thread_safe_state),"get_strategic_graph": lambda thread_safe_state: get_strategic_graph(thread_safe_state=thread_safe_state),
    "generate_scientific_report": lambda dataset, architecture, execute_tool_func, thread_safe_state: generate_scientific_report(dataset=dataset, architecture=architecture, execute_tool_func=execute_tool_func, thread_safe_state=thread_safe_state),
    "generate_finetuning_dataset": lambda execute_tool_func, thread_safe_state: generate_finetuning_dataset(execute_tool_func=execute_tool_func, thread_safe_state=thread_safe_state),
    # Strategic tools
    "flag_architecture": lambda architecture_name, flag, thread_safe_state: flag_architecture(architecture_name=architecture_name, flag=flag, thread_safe_state=thread_safe_state),
    "analyze_strategy_effectiveness": lambda thread_safe_state: analyze_strategy_effectiveness(thread_safe_state=thread_safe_state),
    "propose_intelligent_architecture": lambda agent, thread_safe_state, architecture_filter=None, **kwargs: propose_intelligent_architecture(agent=agent, architecture_filter=architecture_filter, thread_safe_state=thread_safe_state, **kwargs),
    "propose_architecture_upgrade": lambda thread_safe_state, base_file, directive, **kwargs: propose_architecture_upgrade(agent=thread_safe_state.get("agent"), thread_safe_state=thread_safe_state, base_file=base_file, directive=directive, **kwargs),
    "get_optimization_status": lambda thread_safe_state, optimization_threshold=0.95, analysis_window=10: get_optimization_status(thread_safe_state=thread_safe_state, optimization_threshold=optimization_threshold, analysis_window=analysis_window),
    "set_optimization_model": lambda model_name, thread_safe_state: set_optimization_model(model_name=model_name, thread_safe_state=thread_safe_state),
    "run_web_research": lambda query, max_sites, google_search_tool, thread_safe_state: run_web_research(query=query, max_sites=max_sites, google_search_tool=google_search_tool, thread_safe_state=thread_safe_state),
    "query_research_archive": lambda query, thread_safe_state, top_k=3: query_research_archive(query=query, top_k=top_k, thread_safe_state=thread_safe_state),
    "generate_creative_hypotheses": lambda query, thread_safe_state, auto_build=False, **kw: generate_creative_hypotheses(query=query, thread_safe_state=thread_safe_state, auto_build=auto_build),
    "refresh_chronicle": lambda thread_safe_state, **kw: refresh_chronicle(thread_safe_state=thread_safe_state),
    "run_supernet_search": lambda dataset_id, thread_safe_state, max_epochs=50, representation_type=None, gpu_id=None, **kwargs: run_supernet_search(dataset_id=dataset_id, thread_safe_state=thread_safe_state, max_epochs=max_epochs, representation_type=representation_type, gpu_id=gpu_id, **kwargs),
    # Utility tools
    "list_available_tools": lambda thread_safe_state: list_available_tools(thread_safe_state=thread_safe_state),
    "save_session": lambda thread_safe_state: save_session(thread_safe_state=thread_safe_state),
    "create_todo_list": lambda thread_safe_state, tasks=None, content=None: create_todo_list(tasks=tasks, content=content, thread_safe_state=thread_safe_state),
    "read_todo_list": lambda thread_safe_state: read_todo_list(thread_safe_state=thread_safe_state),
    "update_todo_list": lambda task_number, completed, thread_safe_state: update_todo_list(task_number=task_number, completed=completed, thread_safe_state=thread_safe_state),
    "train_final_ensemble_models": lambda base_test_dataset_id, thread_safe_state: train_final_ensemble_models(base_test_dataset_id=base_test_dataset_id, thread_safe_state=thread_safe_state),
    # Emergency Tool
    "stop_application": lambda reason, thread_safe_state: stop_application(reason=reason, thread_safe_state=thread_safe_state),
    # ============================================================
    # IMAGENET COMPETITION TOOLS
    # Available to the agent in imagenet_competition mode.
    # ============================================================
    "run_imagenet_training": lambda thread_safe_state, **kwargs: run_imagenet_training(thread_safe_state=thread_safe_state, **kwargs),
    "run_imagenet_inference": lambda thread_safe_state, **kwargs: run_imagenet_inference(thread_safe_state=thread_safe_state, **kwargs),
    "evaluate_imagenet_predictions": lambda predictions_json_path, ground_truth_json_path, thread_safe_state: evaluate_imagenet_predictions(predictions_json_path=predictions_json_path, ground_truth_json_path=ground_truth_json_path, thread_safe_state=thread_safe_state),
    "generate_kaggle_submission": lambda predictions_json_path, output_csv_path, thread_safe_state: generate_kaggle_submission(predictions_json_path=predictions_json_path, output_csv_path=output_csv_path, thread_safe_state=thread_safe_state),
}


os.makedirs(PERSISTENT_PATHS["custom_architectures_dir"], exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# OPTUNA PERMANENT RESULT LOG — Anti-Repeat Guardrail
# Stores every trial (sampled params + scores) permanently, filtered per
# architecture+manifest so the agent never repeats explored hyperparameter space.
# Only the USER can delete this file — no auto-pruning.
# ──────────────────────────────────────────────────────────────────────────────
OPTUNA_RESULT_LOG_PATH: str = os.path.join(
    PERSISTENT_PATHS.get("processed_data_dir", "processed_data"),
    "optuna_result_log.json",
)
# Coverage threshold: if >75% of the proposed range is already explored, block the sweep.
OVERLAP_BLOCK_THRESHOLD: float = 0.75

# Regularizer params that are unconditionally allowed through the guardrail (Stage 1 safelist).
REGULARIZER_PARAMS: frozenset = frozenset({
    "dropout_rate", "weight_decay", "l2_lambda", "l1_lambda",
    "label_smoothing", "noise_std", "mixup_alpha", "data_augmentation_strength",
})

# Columns that are metadata / targets — never hyperparameters.
# Single source of truth used by analyze_best_vs_worst_trials, get_parameter_importance,
# analyze_hyperparameter_tradeoffs, and _get_and_flatten_log.
METADATA_COLUMNS: frozenset = frozenset({
    "architecture", "dataset_id", "source", "display_str", "index",
    "status", "message", "traceback", "representation_type",
    "manifest_name", "custom_architecture_file", "model_path",
    "mlflow_run_id", "run_id",
    "mean_accuracy", "std_accuracy", "mean_train_accuracy",
})


def get_hyperparameter_columns(df: "pd.DataFrame") -> List[str]:
    """Returns column names that represent hyperparameters (excludes METADATA_COLUMNS)."""
    return [c for c in df.columns if c not in METADATA_COLUMNS]


def _load_optuna_log(architecture: str, manifest_name: str) -> list:
    """
    Loads the permanent Optuna result log and returns only records matching
    the given architecture + manifest_name. Returns [] on any error.
    """
    try:
        if not os.path.exists(OPTUNA_RESULT_LOG_PATH):
            return []
        with open(OPTUNA_RESULT_LOG_PATH, "r", encoding="utf-8") as f:
            all_records: list = json.load(f)
        return [
            r for r in all_records
            if isinstance(r, dict)
            and r.get("architecture") == architecture
            and r.get("manifest_name") == manifest_name
        ]
    except Exception as e:
        log(f"[OptunaLog] Warning: could not load result log: {e}")
        return []


def _append_to_optuna_log(records: list) -> None:
    """
    Appends new trial records to the permanent log. Never overwrites existing entries.
    Each record must contain: timestamp, architecture, manifest_name, experiment_name,
    params (dict of actual sampled values), mean_accuracy, inference_time_ms.
    """
    if not records:
        return
    try:
        existing: list = []
        if os.path.exists(OPTUNA_RESULT_LOG_PATH):
            with open(OPTUNA_RESULT_LOG_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.extend(records)
        with open(OPTUNA_RESULT_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, default=str)
        log(f"[OptunaLog] Appended {len(records)} trial(s). Total log size: {len(existing)}.")
    except Exception as e:
        log(f"[OptunaLog] Warning: could not append to result log: {e}")


def _check_search_space_overlap(
    proposed_hyperparameters: dict,
    past_trials: list,
) -> tuple:
    """
    Pre-flight guardrail: compares the proposed search-space ranges against the
    distribution of previously sampled parameter values.

    Returns (blocked: bool, warning_message: str).
    If blocked is True, the sweep should NOT run — the agent must refine its ranges.

    Coverage per numeric param is measured as:
        fraction of the [proposed_min, proposed_max] range that is "covered" by
        past samples, using a simple bin-fill heuristic (N_BINS=10).
    Categorical params are checked by set-overlap ratio.
    """
    if not past_trials or not proposed_hyperparameters:
        return False, ""

    N_BINS = 10
    covered_params: list = []
    total_params: list = []

    for name, definition in proposed_hyperparameters.items():
        if not isinstance(definition, dict):
            continue
        past_values = [
            t.get("params", {}).get(name)
            for t in past_trials
            if t.get("params", {}).get(name) is not None
        ]
        if not past_values:
            continue

        param_type = definition.get("type", "")
        total_params.append(name)

        if param_type in ("int", "float", "loguniform"):
            p_min = definition.get("min")
            p_max = definition.get("max")
            if p_min is None or p_max is None or p_max <= p_min:
                continue
            # Fill bins
            bin_width = (p_max - p_min) / N_BINS
            filled_bins: set = set()
            for v in past_values:
                try:
                    bin_idx = min(int((float(v) - p_min) / bin_width), N_BINS - 1)
                    if 0 <= bin_idx < N_BINS:
                        filled_bins.add(bin_idx)
                except (TypeError, ValueError):
                    pass
            coverage = len(filled_bins) / N_BINS
            if coverage >= OVERLAP_BLOCK_THRESHOLD:
                covered_params.append(name)

        elif param_type == "choice":
            proposed_choices = set(definition.get("choices", []))
            tested_choices = set(past_values)
            if proposed_choices and len(tested_choices & proposed_choices) / len(proposed_choices) >= OVERLAP_BLOCK_THRESHOLD:
                covered_params.append(name)

    if not total_params:
        return False, ""

    # Stage 1 — Regularizer Safelist: if any regularizer param is uncovered, allow unconditionally.
    proposed_regularizers_uncovered = [
        name for name in proposed_hyperparameters
        if name in REGULARIZER_PARAMS and name not in covered_params
    ]
    if proposed_regularizers_uncovered:
        log(f"[Guardrail] Regularizer params {proposed_regularizers_uncovered} are uncovered — sweep allowed.")
        return False, ""

    # Stage 2 — Quorum: block only when ALL params with past data are covered.
    if len(covered_params) < len(total_params):
        return False, ""

    # Build a human-readable warning with best past result
    uncovered_params = [p for p in total_params if p not in covered_params]
    best_trial = max(past_trials, key=lambda t: t.get("held_out_test_accuracy", 0.0), default={})
    best_params = best_trial.get("params", {})
    best_acc = best_trial.get("held_out_test_accuracy", "N/A")
    best_acc_str = f"{best_acc:.4f}" if isinstance(best_acc, float) else str(best_acc)
    covered_list = ", ".join(f"`{p}`" for p in covered_params)

    warning = (
        f"⚠️ GUARDRAIL: ALL {len(covered_params)}/{len(total_params)} "
        f"params ({covered_list}) are fully covered in this search space for the current architecture+manifest. "
        f"Past best result: accuracy={best_acc_str}, params={best_params}. "
        f"Please refine your search ranges to unexplored regions before calling again."
    )
    return True, warning

# ──────────────────────────────────────────────────────────────────────────────


def stop_application(reason: str, thread_safe_state: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    EMERGENCY TOOL: Stops the entire application immediately.
    Use this ONLY when the agent is stuck in an unrecoverable state or infinite loop.
    """
    log(f"[EMERGENCY STOP] Agent triggered shutdown. Reason: {reason}")
    if thread_safe_state:
        # Optional: Save state before dying?
        pass
    
    # Force exit
    sys.exit(1)
    return {"status": "stopped", "message": "Application stopped."}


def _get_and_flatten_log(
    architecture_filter: Optional[str] = None, thread_safe_state: Optional[Dict[str, Any]] = None
) -> Optional[pd.DataFrame]:
    """
    A helper function to retrieve the results log, flatten the 'params',
    robustly convert types, and then filter. This handles mixed data structures.
    It now accepts an optional thread_safe_state dictionary to retrieve the results_log.
    """
    if thread_safe_state is None:
        return None # Should not happen in proper execution

    results_log = thread_safe_state.get("results_log", [])
    if not results_log:
        log("[Data Helper] Results log is empty.")
        return None

    # Per-cycle cache: avoid re-flattening the same log multiple times in one analyst pass.
    # Invalidated automatically when a new trial is appended (len changes).
    _cache_key = (len(results_log), architecture_filter)
    _cached = thread_safe_state.get("_df_flat_cache")
    if _cached and _cached[0] == _cache_key:
        return _cached[1]

    # --- INIZIO DELLA LOGICA CORRETTA E SICURA ---
    processed_log = []
    for entry in results_log:
        flat_entry = entry.copy()
        if "params" in flat_entry and isinstance(flat_entry.get("params"), dict):
            # Estrai i parametri da appiattire
            params_to_flatten = flat_entry.pop("params")
            for key, value in params_to_flatten.items():
                # CRITICAL: Always inject params, but prefix if they collide with primary keys
                if key in ["architecture", "mean_accuracy", "std_accuracy", "dataset_id", "manifest_name", "experiment_name"]:
                    flat_entry[f"param_{key}"] = value
                else:
                    flat_entry[key] = value
        else:
            pass
        processed_log.append(flat_entry)

    df_flat = pd.DataFrame(processed_log)
    # --- FINE DELLA LOGICA CORRETTA E SICURA ---

    # Filter by architecture (if requested)
    if architecture_filter and "architecture" in df_flat.columns:
        df_flat = df_flat[df_flat["architecture"] == architecture_filter].copy()
        if df_flat.empty:
            log(f"[Data Helper] No trials found for architecture '{architecture_filter}'.")
            return None

    # Robustly convert types to numeric (unchanged)
    meta_cols = [
        "architecture",
        "dataset_id",
        "mean_accuracy",
        "std_accuracy",
        "source",
        "display_str",
        "index",
        "status",
        "message",
        "traceback",
        "representation_type",        # FIX: Protect from numeric coercion
        "manifest_name",              # FIX: Protect from numeric coercion
        "custom_architecture_file",   # FIX: Protect from numeric coercion
    ]
    for col in df_flat.columns:
        if col not in meta_cols:
            df_flat[col] = pd.to_numeric(df_flat[col], errors="coerce")

    thread_safe_state["_df_flat_cache"] = (_cache_key, df_flat)
    return df_flat


# Version: 2.1 (Unified Import Fix + Smart GPU + Stop Tool - 12/02/2026)




def _load_dataset_metadata(thread_safe_state: Dict[str, Any]):
    cfg = thread_safe_state.get("cfg")
    if not os.path.exists(PERSISTENT_PATHS["datasets_metadata"]):
        initial_path = PERSISTENT_PATHS["generated_datasets_dir"]
        os.makedirs(initial_path, exist_ok=True)
        initial_meta = {
            "initial_dataset": {
                "params": "dummy dataset - create your own - do not forget to specify the parameters for the architecures with the manifest",
                "path": initial_path,
            }
        }
        _save_dataset_metadata(initial_meta)
        return initial_meta
    with open(PERSISTENT_PATHS["datasets_metadata"], "r") as f:
        return json.load(f)


def _save_dataset_metadata(metadata):
    os.makedirs(os.path.dirname(PERSISTENT_PATHS["datasets_metadata"]), exist_ok=True)
    with open(PERSISTENT_PATHS["datasets_metadata"], "w") as f:
        json.dump(metadata, f, indent=2)


def list_available_datasets(thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:
    """Lists all available datasets that can be used for training.
    Retrieves metadata using the provided thread-safe state."""
    log("Listing available datasets...")
    return {"status": "completed", "datasets": _load_dataset_metadata(thread_safe_state)}


# In tools.py


def detect_optimal_architecture(data_path: str) -> str:
    """
    Auto-detects optimal architecture based on data shape.
    Loads a sample file from data_path and analyzes shape to recommend best CNN type.
    
    Returns:
        Architecture name (e.g., '2D_IMAGE', '1D_CNN', '2D_SPECTROGRAM')
    """
    try:
        # Find sample data file
        if os.path.isdir(data_path):
            # Look for .npy or .npz files or common image formats
            for root, dirs, files in os.walk(data_path):
                # Check for standard numpy data
                npy_files = [f for f in files if f.endswith(('.npy', '.npz'))]
                if npy_files:
                    sample_path = os.path.join(root, npy_files[0])
                    break
                
                # Check for raw images (folder based)
                img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if img_files:
                    log(f"[Auto-Detect] Found raw image files (e.g., {img_files[0]}). Assuming generic 2D image dataset.")
                    return "2D_IMAGE"
            else:
                log(f"[Auto-Detect] No data files found in {data_path}")
                return "2D_IMAGE"  # Fallback
        else:
            sample_path = data_path
        
        # Load sample to check shape (only if it was npy/npz)
        data = np.load(sample_path)
        
        # Handle both .npy (ndarray) and .npz (archive) formats
        if isinstance(data, np.ndarray):
            shape = data.shape
        else:  # npz file
            arr_name = list(data.files)[0]
            shape = data[arr_name].shape
        
        log(f"[Auto-Detect] Sample data shape: {shape}")
        
        # Map shape → optimal architecture
        if len(shape) == 4:  # (N, H, W, C)
            if shape[-1] == 3:
                log("[Auto-Detect] Detected RGB images (H x W x 3)")
                return "2D_IMAGE"
            elif shape[-1] == 1:
                log("[Auto-Detect] Detected grayscale images (H x W x 1)")
                return "2D_SPECTROGRAM"
            else:
                return "2D_IMAGE"
                
        elif len(shape) == 3:  # (N, H, W) or (N, time, features)
            if shape[1] == shape[2]:  # Square → likely 2D image
                log("[Auto-Detect] Detected 2D square data (spectrogram/heatmap)")
                return "2D_SPECTROGRAM"
            else:
                log("[Auto-Detect] Detected time-series or sequence data")
                return "1D_CNN"
                
        elif len(shape) == 2:  # (N, features)
            log("[Auto-Detect] Detected 1D feature vector")
            return "1D_CNN"
        
        return "2D_IMAGE"  # Fallback
        
    except Exception as e:
        log(f"[Auto-Detect] ERROR during detection: {e}")
        return "2D_IMAGE"  # Safe fallback


# In tools.py

def create_dataset_manifest(manifest_name: str, params: Dict[str, Any], thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a new dataset manifest. 
    AUTO-FIX: Automatically injects default parameters if the agent misses them, 
    based on the detected data type.
    """
    log(f"Creating new dataset manifest: '{manifest_name}'")

    if not re.match(r"^[a-zA-Z0-9_-]+$", manifest_name):
        return {"status": "error", "message": "Invalid manifest_name. Use only letters, numbers, hyphens, and underscores."}

    try:
        metadata = _load_dataset_metadata(thread_safe_state)
        if manifest_name in metadata:
            # Only skip if no representation_type is requested or it is already registered.
            # If the caller requests a new representation_type, fall through to register it.
            requested_rep = params.get("representation_type") if params else None
            already_registered = requested_rep and requested_rep in metadata[manifest_name].get("generated_representations", {})
            if not requested_rep or already_registered:
                return {"status": "completed", "message": f"Manifest '{manifest_name}' already exists and is ready to use. Proceed directly to training."}
            log(f"[Manifest] '{manifest_name}' exists but representation '{requested_rep}' is not registered. Continuing to register it.")

        # --- MANIFEST GATE ---
        # Before creating a brand-new manifest, scan datasets_metadata.json for an
        # equivalent sibling that already has the requested representation generated
        # on disk with identical tensor-defining params. If found, hard-redirect.
        # This blocks both the LLM Decider and the Innovation Team from spawning
        # redundant manifests that would trigger ~30min of data regeneration.
        # See manifest_gate.py for matching rules.
        try:
            from manifest_gate import evaluate_manifest_request
            gate_result = evaluate_manifest_request(
                requested_name=manifest_name,
                requested_params=params or {},
                metadata=metadata,
                thread_safe_state=thread_safe_state,
            )
            if gate_result is not None:
                # Gate decided: either redirect to existing sibling or halt for user.
                return gate_result
        except Exception as gate_err:
            # Defensive: never let the gate crash manifest creation. Log and continue.
            log(f"[Manifest] WARNING: gate evaluation failed ({gate_err}). Proceeding with normal creation.")

        # --- AUTO-DETECTION & DEFAULT INJECTION ---
        # 1. Identify Data Source
        source_name = thread_safe_state.get("selected_raw_data_source")
        raw_data_dir = PERSISTENT_PATHS.get("raw_data_dir", "processed_data/raw_data")
        source_path = os.path.join(raw_data_dir, source_name) if source_name else ""

        # 2. Detect Type (Image vs Timeseries)
        detected_type = "2D_GENERIC_IMAGE" # Fallback

        # Known representation families (order matters: longer/more specific first)
        _KNOWN_REPS = [
            "3D_DYNAMIC_GAF", "3D_DYNAMIC_CWT", "3D_WAVELET_CWT",
            "3D_GAF_VIDEO", "3D_VIDEO",
            "2D_CWT_SCALOGRAM", "2D_SPECTROGRAM", "2D_GAF",
            "1D_CNN",
        ]

        # Check if config overrides detection
        cfg_selected_model = thread_safe_state.get("cfg", {}).get("optimization", {}).get("selected_model")
        if params.get("representation_type"):
            detected_type = params.get("representation_type")
            log(f"[Manifest] Agent explicitly hinted representation_type '{detected_type}'. Using it as primary directive.")
        else:
            # Infer from manifest name before falling back to cfg_selected_model —
            # cfg reflects the current optimization focus, not the manifest's representation.
            manifest_name_upper = manifest_name.upper()
            inferred = next((r for r in _KNOWN_REPS if r in manifest_name_upper), None)
            if inferred:
                detected_type = inferred
                log(f"[Manifest] Inferred representation '{detected_type}' from manifest name '{manifest_name}'.")
            else:
                # No representation could be determined — reject instead of silently falling back.
                # A silent fallback to cfg_selected_model causes wrong routing and wastes compute.
                return {
                    "status": "error",
                    "message": (
                        f"[Manifest ERROR] Cannot determine representation_type for manifest '{manifest_name}'. "
                        f"You MUST pass 'representation_type' explicitly in params (e.g. '2D_GAF', '1D_CNN', '3D_DYNAMIC_GAF'). "
                        f"Do NOT rely on auto-detection. Re-call create_dataset_manifest with the correct representation_type."
                    ),
                }
        
        # 3. Inject Defaults if missing
        defaults = {}
        timeseries_2d_types = ["2D_GAF", "2D_SPECTROGRAM", "2D_CWT_SCALOGRAM", "2D_CWT"]
        if detected_type in timeseries_2d_types or "1D" in detected_type or "VIDEO" in detected_type:
            # Defaults for Timeseries/Audio (even if represented as 2D)
            defaults = {
                "n_fft": 512,
                "hop_length": 64,
                "gaf_image_size": 48,
                "video_num_segments": 8,
                "video_gaf_image_size": 24
            }
        elif "2D" in detected_type or "IMAGE" in detected_type or "cifar" in manifest_name.lower():
            # Defaults for Images (Specific CIFAR-10 optimization)
            if "cifar" in manifest_name.lower() or "cifar" in detected_type.lower():
                log("[Manifest] Detected CIFAR dataset context. Applying standard CIFAR-10 parameters (32x32, 3ch).")
                defaults = {
                    "image_size": [32, 32], 
                    "channels": 3,
                    "num_classes": 10,
                    "use_data_augmentation": True,
                    "batch_size": 64,
                    "info": "Auto-configured standard CIFAR-10 parameters"
                } 
            else:
                # Generic Images
                defaults = {
                    "image_size": [64, 64], 
                    "channels": 3,
                    "use_data_augmentation": True,
                    "batch_size": 32
                }
        
        # Merge: User params overwrite defaults, but defaults fill gaps
        final_params = {**defaults, **params}
        
        # Ensure raw_data_path is set for the worker
        if "raw_data_path" not in final_params and source_path:
             final_params["raw_data_path"] = source_path

        # ------------------------------------------

        # Create basic manifest structure
        manifest_path = os.path.join(PERSISTENT_PATHS["generated_datasets_dir"], manifest_name)
        
        metadata[manifest_name] = {
            "params": final_params, # Save the robust params
            "path": manifest_path,
            "generated_representations": {},
            "recommended_architecture": detected_type
        }

        _save_dataset_metadata(metadata)
        os.makedirs(manifest_path, exist_ok=True)

        log(f"Successfully created manifest '{manifest_name}' with params: {final_params}")
        
        # --- AUTO-MEMORIZE DATASET CREATION ---
        try:
            finding_text = f"[DATASET CREATED] Source: '{source_name}' | Manifest: '{manifest_name}' | Arch: {detected_type} | Params: {final_params}"
            memorize_finding(
                finding=finding_text,
                experiment_name="dataset_creation",
                thread_safe_state=thread_safe_state,
                manifest_name=manifest_name,
                representation_type=detected_type
            )
            log(f"[Manifest] Auto-memorized dataset creation for '{manifest_name}'.")
        except Exception as e:
            log(f"[Manifest WARNING] Failed to auto-memorize dataset creation: {e}")
        # --------------------------------------

        return {
            "status": "completed",
            "manifest_name": manifest_name,
            "message": f"Manifest created successfully. Auto-detected type: {detected_type}. Defaults applied for missing params.",
            "recommended_architecture": detected_type,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}



def generate_representation(manifest_name: str, representation_type: str, thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a specific data representation and robustly reports errors from the worker.
    """
    log(f"Generating representation '{representation_type}' for manifest '{manifest_name}'...")

    try:
        metadata = _load_dataset_metadata(thread_safe_state)
        if manifest_name not in metadata:
            return {"status": "error", "message": f"Manifest '{manifest_name}' not found."}

        manifest = metadata[manifest_name]
        params = manifest.get("params", {})



        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            result_path = tmp.name

        selected_source = thread_safe_state.get("selected_raw_data_source")
        if not selected_source:
             # Fallback to the path stored in the manifest params
             selected_source = params.get("raw_data_path")
             if not selected_source:
                 return {"status": "error", "message": "No raw data source selected and no 'raw_data_path' in manifest params."}
             log(f"Using raw data source from manifest params: {selected_source}")

        log(f"Using '{selected_source}' as the raw data source for generation.")

        # Celery Task Execution
        task_result = run_data_pipeline_worker.apply_async(
            args=[selected_source, manifest_name, representation_type, params, result_path]
        )
        
        try:
            # 3D representations (CWT, dynamic GAF) on large datasets can take 30-60 min.
            # Timeout increased to 48h to handle massive datasets on local GPUs.
            process_output = task_result.get(timeout=3600 * 48)
        except Exception as e:
            return {"status": "error", "message": f"Celery task timed out or failed: {e}"}

        if process_output.get("status") == "error":
             log(f"[WORKER CRASH] The data generation worker failed. Stderr:\n{process_output.get('stderr')}\nTraceback:\n{process_output.get('traceback')}")
             return {
                "status": "error",
                "message": "The data generation worker script crashed.",
                "worker_error": process_output.get('stderr'),
                "traceback": process_output.get("traceback", ""),
                "error_details": process_output.get("error", process_output.get("message", ""))
            }

        # process_output already contains the worker's JSON result (returned by _run_subprocess_task)
        result = process_output

        if result.get("status") == "completed":
            data_file_path = result["data_file_path"]
            if "generated_representations" not in metadata[manifest_name]:
                metadata[manifest_name]["generated_representations"] = {}
            metadata[manifest_name]["generated_representations"][representation_type] = data_file_path

            # Register finetune split if the data pipeline auto-generated it
            if result.get("has_finetune_split"):
                if "generated_finetune_representations" not in metadata[manifest_name]:
                    metadata[manifest_name]["generated_finetune_representations"] = {}
                manifest_dir = metadata[manifest_name].get("path") or os.path.dirname(data_file_path)
                ft_path = os.path.join(manifest_dir, f"{representation_type}_X_finetune.npy")
                metadata[manifest_name]["generated_finetune_representations"][representation_type] = ft_path
                metadata[manifest_name]["has_finetune_split"] = True
                log(f"[Finetune] Registered '{representation_type}' finetune split -> {ft_path}")

            # Register shifted-test split if auto-generated
            if result.get("has_shifted_test"):
                if "generated_shifted_test_representations" not in metadata[manifest_name]:
                    metadata[manifest_name]["generated_shifted_test_representations"] = {}
                manifest_dir = metadata[manifest_name].get("path") or os.path.dirname(data_file_path)
                sh_path = os.path.join(manifest_dir, f"{representation_type}_X_shifted_test.npy")
                metadata[manifest_name]["generated_shifted_test_representations"][representation_type] = sh_path
                metadata[manifest_name]["has_shifted_test"] = True
                log(f"[ShiftedTest] Registered '{representation_type}' shifted test set -> {sh_path}")

            _save_dataset_metadata(metadata)
            log(f"Successfully generated and registered '{representation_type}' for '{manifest_name}'.")
            return {
                "status": "completed",
                "manifest_name": manifest_name,
                "message": f"{representation_type} data generated.",
                "has_finetune_split": bool(result.get("has_finetune_split")),
                "has_shifted_test":  bool(result.get("has_shifted_test")),
            }
        else:
            return result

    except Exception:
        return {"status": "error", "message": f"Data generation orchestration failed: {traceback.format_exc()}"}
    finally:
        if "result_path" in locals() and os.path.exists(result_path):
            os.remove(result_path)


def evaluate_ensemble(dataset_id: str, thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    delegates the final ensemble evaluation to a dedicated worker process."""
    log("Delegating final ensemble evaluation to a dedicated worker...")
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        result_path = tmp.name
    try:
        task_result = run_ensemble_evaluation_worker.apply_async(
            args=[dataset_id, result_path]
        )
        # Wait for result
        process_output = task_result.get(timeout=600)
        
        if process_output.get("status") == "error":
             result = {"status": "error", "message": f"Ensemble evaluation worker failed: {process_output.get('stderr')}"}
        else:
            with open(result_path, "r") as f:
                result = json.load(f)
    except Exception as e:
        result = {"status": "error", "message": f"Celery task failed: {e}"}
    finally:
        if os.path.exists(result_path):
            os.remove(result_path)
    return result


def memorize_finding(finding: str, experiment_name: str, thread_safe_state: Dict[str, Any], manifest_name: Optional[str] = None, representation_type: Optional[str] = None, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Saves a finding to the unified long-term memory and returns its unique ID.
    Auto-recovers context (manifest, representation) from logs if not provided.
    Injects Raw Data Source and complete Manifest Params into the memory string for LLM awareness.
    """
    try:
        log(f"[MEMORY] Memorizing finding for '{experiment_name}'...")
        
        # --- MISSION 3: AUTO-RECOVERY ---
        if not manifest_name or not representation_type:
            results_log = thread_safe_state.get("results_log", [])
            if results_log:
                last_trial = results_log[-1]
                if not manifest_name:
                    manifest_name = last_trial.get("manifest_name", "unknown_manifest")
                    log(f"[MEMORY] Auto-recovered manifest_name: {manifest_name}")
                if not representation_type:
                    representation_type = last_trial.get("representation_type", "unknown_rep")
                    log(f"[MEMORY] Auto-recovered representation_type: {representation_type}")
        # --------------------------------

        finding_id = str(uuid.uuid4())

        selected_source = thread_safe_state.get("selected_raw_data_source")
        if not selected_source:
             # Try to recover source from datasets_metadata if manifest_name exists
             try:
                 md = _load_dataset_metadata(thread_safe_state)
                 if manifest_name and manifest_name in md:
                     selected_source = md[manifest_name].get("params", {}).get("raw_data_path", "unknown")
                     log(f"[MEMORY] Recovered source from manifest: {selected_source}")
             except:
                 pass
             if not selected_source:
                 selected_source = "unknown"

        data_type = "unknown"
        data_info = "No description available."
        try:
            metadata_path = os.path.join(PERSISTENT_PATHS["raw_data_dir"], selected_source, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                data_type = metadata.get("data_nature", "unknown")
                data_info = metadata.get("description", "No description available.")
        except Exception as e:
            log(f"[MEMORY WARNING] Could not load metadata for '{selected_source}'. Using defaults. Error: {e}")

        # --- DATASET METADATA INJECTION ---
        manifest_params_str = "{}"
        try:
            md = _load_dataset_metadata(thread_safe_state)
            if manifest_name and manifest_name in md:
                # We exclude raw_data_path from params to avoid duplication, keeping only config params
                params_dict = {k: v for k, v in md[manifest_name].get("params", {}).items() if k != "raw_data_path"}
                manifest_params_str = json.dumps(params_dict)
        except Exception as e:
             log(f"[MEMORY WARNING] Failed to load parameters for '{manifest_name}': {e}")
        
        # Prepend rich context so the LLM always sees what dataset/manifest this finding belongs to.
        # C1: Truncate params JSON to 200 chars to avoid diluting the semantic signal.
        # Skip prepend entirely if manifest is unknown (no informational value).
        if len(manifest_params_str) > 200:
            manifest_params_str = manifest_params_str[:197] + "..."
        category_tag = f" | Category: {category}" if category else ""
        rich_context = f"[Dataset: {selected_source} | Manifest: {manifest_name} | Type: {representation_type}{category_tag} | Params: {manifest_params_str}]"

        # Don't duplicate context if it's the auto-memorization from create_dataset_manifest
        # and skip prepend when manifest_name is unknown (no value added)
        if "[DATASET CREATED]" not in finding and manifest_name not in ("unknown_manifest", "unknown", None):
            finding = f"{rich_context} {finding}"

        # Append context to data_info for database querying
        data_info += f" | {rich_context}"
        # ----------------------------------

        memory_instance = thread_safe_state["memory"]
        memory_instance.add(
            finding=finding,
            source_experiment=experiment_name,
            finding_id=finding_id,
            data_source_id=selected_source,
            data_type=data_type,
            data_info=data_info,
        )

        return {
            "status": "completed",
            "message": "Finding successfully memorized with full data context.",
            "finding_id": finding_id,
        }
    except Exception as e:
        return {"status": "error", "message": f"Failed to memorize finding: {e}"}


def recall_relevant_memories(query: str, thread_safe_state: Dict[str, Any], k: int = 5, summarize: bool = True) -> Dict[str, Any]:
    """
    Searches the entire long-term memory to find relevant past experiences across all experiments.
    
    **TOKEN OPTIMIZATION:** If summarize=True (default), raw findings are condensed via LLM 
    to reduce token consumption (typically 80-90% reduction).
    """
    try:
        log(f"[MEMORY] Recalling memories from global store with query: '{query}'")

        current_source_id = thread_safe_state.get("selected_raw_data_source")
        current_data_type = "unknown"
        
        # Attempt to load metadata to get data context
        if current_source_id:
            try:
                metadata_path = os.path.join(PERSISTENT_PATHS["raw_data_dir"], current_source_id, "metadata.json")
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                current_data_type = metadata.get("data_nature", "unknown")
            except Exception:
                pass

        # Perform the search
        memory_instance = thread_safe_state["memory"]
        recalled_memories = memory_instance.search(
            query, current_data_source_id=current_source_id, current_data_type=current_data_type, k=k
        )

        count = len(recalled_memories) if recalled_memories else 0
        log(f"[MEMORY] Found {count} relevant memories.")

        if not recalled_memories:
            return {"status": "completed", "memories": "No relevant memories found in the global store."}

        # Create a version of the memories without the embedding for the agent
        findings_for_agent = [
            f"[From experiment: {mem['source_experiment']}] {mem['finding']}" for mem in recalled_memories
        ]

        # --- TOKEN OPTIMIZATION: LLM Summarization ---
        if summarize:
            try:
                # Check if agent is available
                if "agent" not in thread_safe_state:
                    log("[MEMORY] Warning: Agent not in thread_safe_state, skipping summarization. Returning raw findings.")
                else:
                    agent = thread_safe_state["agent"]
                    
                    # Concatenate raw findings
                    raw_findings_text = "\n\n".join(findings_for_agent)
                    
                    log(f"[MEMORY] Summarizing {len(raw_findings_text)} chars of raw memories...")
                    
                    # Call agent's SYNCHRONOUS summarization method
                    summarized_findings = agent.summarize_long_term_memory(raw_findings_text)
                    
                    log(f"[MEMORY] Summarized to {len(summarized_findings)} chars (reduction: {100*(1-len(summarized_findings)/len(raw_findings_text)):.1f}%)")
                    
                    # Replace raw findings with summary
                    findings_for_agent = summarized_findings
                    
            except Exception as e:
                log(f"[MEMORY] Summarization failed, returning raw findings. Error: {e}")
                import traceback
                log(f"[MEMORY] Traceback: {traceback.format_exc()}")
                # Fall back to raw findings if summarization fails
        # -----------------------------------------------

        # Create a clean list of memories for the state update, also without embeddings
        memories_for_state = []
        for mem in recalled_memories:
            clean_mem = mem.copy()
            clean_mem.pop('embedding', None)  # Safely remove the embedding
            memories_for_state.append(clean_mem)

        # --- CRITICAL FIX: Return summarized findings in BOTH fields ---
        # The "new_last_memory_retrieval" should contain the summary, not the raw dump
        return {
            "status": "completed",
            "memories": findings_for_agent,  # Summarized (if enabled)
            "new_last_memory_retrieval": {
                "query": query, 
                "summary": findings_for_agent if isinstance(findings_for_agent, str) else "\n".join(findings_for_agent),
                "raw_count": len(recalled_memories),
                "summarized": summarize
            },
        }

    except Exception as e:
        # Catch any errors and return them so the agent knows something went wrong
        return {"status": "error", "message": f"Failed to recall memories: {str(e)}"}

# Ensure this import is at the top of tools.py

from tools import generate_representation  # Importa la funzione di generazione interna

def run_training_trial(
    thread_safe_state: Dict[str, Any], 
    manifest_name: str = "trial_auto",
    representation_type: Optional[str] = None,
    experiment_name: Optional[str] = None,
    custom_architecture_file: Optional[str] = None,
    save_model: bool = False,
    **hyperparameters
) -> Dict[str, Any]:
    """
    Esegue un trial di training delegandolo a un worker.
    Include:
    - Auto-generazione dei dati se mancanti.
    - Configurazione sicura per GPU A100 (XLA disabilitato/limitato).
    - Salvataggio immediato dei risultati su disco.
    """
    # Determine GPU ID safely
    # If gpu_id is passed as a top-level kwarg (from agent), use it.
    # Otherwise check hyperparameters.
    gpu_id = None
    if "gpu_id" in hyperparameters:
        gpu_id = hyperparameters.pop("gpu_id") # Extract and remove from metrics to avoid confusion
    
    # Fallback/Override logic could go here if needed
        
    log(f"Received training trial request. Manifest: '{manifest_name}'. Rep: '{representation_type}'. GPU: {gpu_id}")
    log(f"[Tool Debug] Hyperparameters received: {hyperparameters}") # Added for debugging

    # 1. Validazione Argomenti
    VALID_REPRESENTATION_TYPES = [
        "1D_CNN", "2D_SPECTROGRAM", "2D_GAF", "2D_CWT_SCALOGRAM",
        "2D_GENERIC_IMAGE", "2D_IMAGE", "3D_VIDEO", "3D_GAF_VIDEO",
        "3D_DYNAMIC_GAF", "3D_DYNAMIC_CWT", "3D_WAVELET_CWT"
    ]

    try:
        metadata = _load_dataset_metadata(thread_safe_state)
        # Check manifest existence first to give better hints
        available_reps = []
        if manifest_name in metadata:
            available_reps = list(metadata[manifest_name].get("generated_representations", {}).keys())
    except:
        available_reps = []

    if custom_architecture_file and not representation_type:
        msg = f"Argument 'representation_type' is MANDATORY when using 'custom_architecture_file'. "
        if available_reps:
            msg += f"Available data for '{manifest_name}': {available_reps}. "
        else:
            msg += f"No data generated yet for '{manifest_name}'. "
        msg += f"Valid options to generate: {VALID_REPRESENTATION_TYPES}"
        return {"status": "error", "message": msg}

    if representation_type and representation_type not in VALID_REPRESENTATION_TYPES:
         # Check if it is an alias or just a typo/hallucination
         return {
             "status": "error", 
             "message": f"Invalid representation_type '{representation_type}'. Must be one of: {VALID_REPRESENTATION_TYPES}"
         }

    # 1.5b. Block incompatible custom arch + 2D representation combinations
    _3D_ONLY_ARCHS = {"hybrid_resnet_gaf_ltc.py"}
    _3D_REPRESENTATIONS = {"3D_DYNAMIC_GAF", "3D_DYNAMIC_CWT", "3D_WAVELET_CWT", "3D_VIDEO", "3D_GAF_VIDEO"}
    if custom_architecture_file in _3D_ONLY_ARCHS and representation_type and representation_type not in _3D_REPRESENTATIONS:
        return {
            "status": "error",
            "message": (
                f"[INVALID COMBINATION] '{custom_architecture_file}' requires a 3D representation "
                f"(N_frames, H, W, C) but got '{representation_type}' which is 2D. "
                f"Use one of: {sorted(_3D_REPRESENTATIONS)}"
            )
        }

    # 1.6. Missing hyperparameter warning — non-fatal but surfaced in result
    _REQUIRED_HYPERPARAMS = [
        "learning_rate", "dropout_rate", "filters", "kernel_size",
        "batch_size", "epochs", "patience", "weight_decay", "batch_norm"
    ]
    _missing_params = [p for p in _REQUIRED_HYPERPARAMS if p not in hyperparameters]
    if _missing_params:
        log(f"[Tool Warning] run_training_trial missing params (worker will use silent defaults): {_missing_params}")

    # 1.5. Pre-flight check file existence to prevent cluster freeze
    if custom_architecture_file:
        __arch_path = os.path.join(PERSISTENT_PATHS["custom_architectures_dir"], custom_architecture_file)
        if not os.path.exists(__arch_path) and not os.path.exists(__arch_path + ".py"):
            return {
                "status": "error",
                "message": f"[ERRORE AGENTE]: Il file '{custom_architecture_file}' non e' stato trovato in '{PERSISTENT_PATHS['custom_architectures_dir']}'. Hai dimenticato di crearlo con 'write_architecture_file'?"
            }

    effective_representation = representation_type
    
    if not effective_representation and not custom_architecture_file:
        # Try to infer from manifest availability
        try:
            m_data = _load_dataset_metadata(thread_safe_state).get(manifest_name, {})
            avail = list(m_data.get("generated_representations", {}).keys())
            if avail:
                effective_representation = avail[0]
                log(f"[Auto-Select] No representation specified. Using available: '{effective_representation}'")
        except:
            pass
            
    if not effective_representation:
         effective_representation = experiment_name or "default_experiment"

    # FORCE HOMOGENIZATION: If we ended up with the legacy name, upgrade it now.
    if effective_representation == "2D_GENERIC_IMAGE":
        effective_representation = "2D_IMAGE"
        log(f"[Auto-Fix] Upgraded legacy representation name '2D_GENERIC_IMAGE' to '2D_IMAGE'.")

    # 2. Controllo e Preparazione Dati (Con Auto-Fix)
    try:
        metadata = _load_dataset_metadata(thread_safe_state)
        
        # Se il manifest non esiste proprio, prova a crearlo automaticamente.
        if manifest_name not in metadata:
            log(f"[Auto-Recovery] Manifest '{manifest_name}' not found. Attempting to create it with default parameters...")
            try:
                # Create with parameters to trigger correct auto-defaults in create_dataset_manifest
                create_dataset_manifest(manifest_name, {"representation_type": effective_representation}, thread_safe_state)
                # Reload metadata to confirm creation
                metadata = _load_dataset_metadata(thread_safe_state)
                if manifest_name not in metadata:
                     return {"status": "error", "message": f"[SYSTEM_ERROR] Manifest '{manifest_name}' not found and auto-creation failed."}
                log(f"[Auto-Recovery] Manifest '{manifest_name}' successfully created.")
            except Exception as e:
                return {"status": "error", "message": f"[SYSTEM_ERROR] Manifest '{manifest_name}' not found and auto-creation crashed: {e}"}
        
        manifest_data = metadata[manifest_name]
        
        # --- AUTO-FIX: Se i dati mancano O il file non esiste, generali ora! ---
        needs_generation = False

        if effective_representation not in manifest_data.get("generated_representations", {}):
            needs_generation = True
        else:
            # Check physical file existence
            file_info = manifest_data["generated_representations"][effective_representation]
            # Handle both string path (old format) and dict (new format)
            file_path = file_info if isinstance(file_info, str) else file_info.get("path")

            # Smart Path Check: Handle cross-platform absolute path mismatches
            # 1. Check the path exactly as stored in metadata
            file_exists = file_path and os.path.exists(file_path)

            # 2. If not found, check the expected local path dynamically
            if not file_exists and file_path:
                expected_local_path = os.path.join(PERSISTENT_PATHS["generated_datasets_dir"], manifest_name, f"{effective_representation}_data.npy")
                if os.path.exists(expected_local_path):
                    log(f"[Path Fix] Ignoring metadata path mismatch. Found local file at: {expected_local_path}")
                    file_exists = True

            if not file_exists:
                log(f"[Auto-Fix] Metadata says data exists, but file '{file_path}' is missing (and not found locally). Forcing regeneration.")
                needs_generation = True

        # --- AUTO-FIX: ancillary splits (shifted_test, finetune) ---
        # If the source raw_data has a shifted_test_folder or finetune source but the
        # corresponding generated file is missing, force regeneration so the
        # data pipeline's auto-discovery code (added later) populates them.
        if not needs_generation:
            try:
                source_root = manifest_data.get("params", {}).get("raw_data_path") or thread_safe_state.get("selected_raw_data_source")
                manifest_dir = os.path.join(PERSISTENT_PATHS["generated_datasets_dir"], manifest_name)
                if source_root:
                    # Check shifted_test
                    has_shifted_src = os.path.exists(os.path.join(source_root, "shifted_test_folder"))
                    has_shifted_gen = os.path.exists(os.path.join(manifest_dir, f"{effective_representation}_X_shifted_test.npy"))
                    if has_shifted_src and not has_shifted_gen:
                        log(f"[Auto-Fix] shifted_test_folder exists in source but '{effective_representation}_X_shifted_test.npy' is missing. Forcing regeneration.")
                        needs_generation = True
                    # Check finetune (any known location)
                    finetune_src_present = any(
                        os.path.exists(os.path.join(source_root, p, "X_finetune.npy"))
                        for p in ("", "finetune_data_folder", "test_data_folder_clinical")
                    )
                    has_finetune_gen = os.path.exists(os.path.join(manifest_dir, f"{effective_representation}_X_finetune.npy"))
                    if finetune_src_present and not has_finetune_gen:
                        log(f"[Auto-Fix] X_finetune.npy exists in source but '{effective_representation}_X_finetune.npy' is missing. Forcing regeneration.")
                        needs_generation = True
            except Exception as _check_e:
                log(f"[Auto-Fix Warning] ancillary-splits check failed: {_check_e}")

        if needs_generation:
            log(f"[Auto-Fix] Data for '{effective_representation}' missing or not found in '{manifest_name}'. Generating automatically...")
            
            # Chiama la funzione di generazione (definita nello stesso file tools.py)
            gen_result = generate_representation(manifest_name, effective_representation, thread_safe_state)
            
            if gen_result.get("status") != "completed":
                detailed_msg = gen_result.get('message', 'Unknown error')
                if gen_result.get('worker_error'):
                    detailed_msg += f" | Details: {gen_result.get('worker_error')}"
                return {"status": "error", "message": f"[SYSTEM_ERROR] Auto-generation failed: {detailed_msg}"}
            
            # Ricarica i metadati perché sono stati aggiornati dalla generazione
            metadata = _load_dataset_metadata(thread_safe_state)
            manifest_data = metadata[manifest_name]
            log(f"[Auto-Fix] Data generated successfully. Proceeding to training.")
        # --------------------------------------------------
            
        # Use dynamic path calculation to ensure it matches current environment (Host vs Docker)
        data_path_root = os.path.join(PERSISTENT_PATHS["generated_datasets_dir"], manifest_name)

    except Exception as e:
         return {"status": "error", "message": f"[SYSTEM_ERROR] Metadata/Data check failed: {str(e)}"}

    # 3. Esecuzione del Worker di Training
    try:
        # Crea un file temporaneo per catturare il risultato JSON del worker
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', dir=PERSISTENT_PATHS["processed_data_dir"]) as tmp:
            result_path = tmp.name

        # --- CONFIGURATION VIA CELERY ---
        # Note: GPU management is now handled by the Celery Worker (which queue it pulls from)
        # or logic inside the task. For now, we dispatch to general queue.
        # Future: Use routing_key='gpu' to send to GPU workers.
        
        # Prepare params for worker
        # Ensure representation_type is passed so worker finds the right data file
        # regardless of what the experiment_name is.
        hyperparameters["representation_type"] = effective_representation
        # Pass the save_model flag
        hyperparameters["save_model"] = save_model
        
        # FIX: Ensure a reasonable default for epochs if missing
        if "epochs" not in hyperparameters:
            hyperparameters["epochs"] = 300
            log(f"[Tool Auto-Fix] 'epochs' not specified. Setting default to 300 (Relies on EarlyStopping).")
        
        # Retrieve test_data_folder and training overrides from config if available
        test_data_folder = None
        cfg = thread_safe_state.get("cfg")
        if cfg:
             # handle both dict and OmegaConf/object
             data_cfg = cfg.get("data") if isinstance(cfg, dict) else getattr(cfg, "data", None)
             if data_cfg:
                 if isinstance(data_cfg, dict):
                     test_data_folder = data_cfg.get("test_data_folder")
                 elif hasattr(data_cfg, "test_data_folder"):
                     test_data_folder = data_cfg.test_data_folder

             # Inject training config defaults (only if the agent did not pass them explicitly)
             training_cfg = cfg.get("training") if isinstance(cfg, dict) else getattr(cfg, "training", None)
             if training_cfg:
                 _get = (lambda k, d: training_cfg.get(k, d)) if isinstance(training_cfg, dict) else (lambda k, d: getattr(training_cfg, k, d))
                 if "patience" not in hyperparameters:
                     hyperparameters["patience"] = _get("early_stopping_patience", 20)
                     log(f"[Config] early_stopping_patience={hyperparameters['patience']} (from config.yaml)")
                 if "checkpoint_every_n_epochs" not in hyperparameters:
                     ckpt = _get("checkpoint_every_n_epochs", 0)
                     if ckpt:
                         hyperparameters["checkpoint_every_n_epochs"] = ckpt
                         log(f"[Config] checkpoint_every_n_epochs={ckpt} (from config.yaml)")

        log(f"Dispatching training task via Celery (Exp: {experiment_name})...")

        task_result = run_training_worker.apply_async(
            args=[experiment_name, hyperparameters, result_path],
            kwargs={
                "data_path_root": data_path_root,
                "custom_architecture_file": custom_architecture_file,
                "test_data_folder": test_data_folder,
                "gpu_id": gpu_id  # <--- PASS GPU ID TO CELERY TASK
            }
        )
        
        # PROGRESS LOGGING - Show task ID and monitoring info
        log(f"[Training] Task dispatched: ID={task_result.id}")
        log(f"[Training] Monitor: tail -f logs/celery_worker_*.log | grep {task_result.id}")
        log(f"[Training] Output file: {result_path}")
        
        start_time = time.time()
        max_wait_seconds = 3600 * 48  # Increased from 4h to 48h for heavy local training
        poll_interval = 30  # seconds
        last_update = start_time
        
        try:
            # Replaced polling loop with blocking wait to reduce log noise
            log(f"[Training] Waiting for task completion (Timeout: {int(max_wait_seconds/3600)}h)...")
            process_output = task_result.get(timeout=max_wait_seconds)
            
        except Exception as e:
            elapsed_mins = int((time.time() - start_time) / 60)
            log(f"[Training] FAILED or TIMEOUT after {elapsed_mins} minutes: {str(e)}")
            if "Timeout" in str(e) or isinstance(e, TimeoutError):
                 task_result.revoke(terminate=True)
            raise subprocess.CalledProcessError(1, ["celery_task"], stderr=str(e))
            
        except Exception as e:
            elapsed_mins = int((time.time() - start_time) / 60)
            log(f"[Training] FAILED after {elapsed_mins} minutes: {str(e)}")
            raise subprocess.CalledProcessError(1, ["celery_task"], stderr=str(e))

        end_time = time.time()
        duration = end_time - start_time
        mins = int(duration / 60)
        secs = int(duration % 60)
        log(f"[Training] Task finished in {mins}m {secs}s")
        
        if process_output.get("status") == "error":
            error_msg = process_output.get('message', 'Unknown worker error')
            worker_log = process_output.get('stderr', '') or process_output.get('stdout', '')
            log(f"[Training] Worker reported ERROR: {error_msg}")
            # Raise with full context in stderr so the outer try/except can capture it
            raise subprocess.CalledProcessError(1, ["celery_task"], stderr=f"{error_msg}\n\nWORKER LOG:\n{worker_log}")

        # [Debug] Check for GPU confirmation in worker output
        if process_output.get("stdout"):
            for line in process_output["stdout"].splitlines():
                if "TF Memory Growth Enabled" in line or "Physical GPUs found" in line:
                     log(f"[Worker Output] {line.strip()}")
        
        # Leggi il risultato
        with open(result_path, 'r') as f:
            result = json.load(f)

        # 4. Salvataggio Immediato dei Risultati (Real-Time Logging)
        if result.get("status") == "completed":
            # Arricchisci il risultato con i metadati del trial
            result["architecture"] = custom_architecture_file if custom_architecture_file else experiment_name
            result["manifest_name"] = manifest_name
            # REPRODUCIBILITY FIX: Use effective params (with defaults) if worker reported them
            result["params"] = result.get("effective_hyperparameters", hyperparameters)
            
            # --- BENCHMARK LOGGING ---
            try:
                from benchmark_logger import log_benchmark_result
                log_benchmark_result(
                    experiment_id=f"trial_{experiment_name}_{int(time.time())}",
                    method="StandardTrial",
                    dataset=manifest_name,
                    accuracy=result.get("mean_accuracy", 0.0),
                    duration_seconds=duration,
                    gpu_hours=duration / 3600.0,
                    inference_time_ms=result.get("inference_time_ms", 0.0),
                    params_count=result.get("params_count", 0),
                    model_size_mb=result.get("model_size_mb", 0.0),
                    params=hyperparameters
                )
            except ImportError:
                log("[Benchmark] Warning: benchmark_logger not found.")
            # -------------------------
            result["source"] = "agent_trial"
            result["missing_params_warning"] = _missing_params if _missing_params else None

            # Aggiungi alla lista nello stato condiviso
            if "results_log" in thread_safe_state:
                thread_safe_state["results_log"].append(result)

                # SALVA SU DISCO ORA!
                save_state(thread_safe_state)
                log(f"[Real-Time Save] Results log successfully saved to disk for {result['architecture']}.")

                # --- LOG METRICHE A MLFLOW ---
                try:
                    from cluster.mlflow_utils import log_agent_metrics, log_agent_artifact
                    import mlflow
                    run_id = thread_safe_state.get("current_mlflow_run_id")
                    if run_id and not run_id.startswith("offline_run_"):
                        metrics = {}
                        if result.get("mean_accuracy") is not None:
                            metrics["mean_accuracy"] = float(result["mean_accuracy"])
                        if result.get("held_out_accuracy") is not None:
                            metrics["held_out_accuracy"] = float(result["held_out_accuracy"])
                        if result.get("inference_time_ms") is not None:
                            metrics["inference_time_ms"] = float(result["inference_time_ms"])
                        if result.get("params_count") is not None:
                            metrics["params_count"] = float(result["params_count"])
                        if result.get("model_size_mb") is not None:
                            metrics["model_size_mb"] = float(result["model_size_mb"])
                        step = len(thread_safe_state.get("results_log", []))
                        if metrics:
                            log_agent_metrics(run_id, metrics, step=step)
                        # Loga anche i parametri usati
                        active = mlflow.active_run()
                        if active and active.info.run_id == run_id:
                            flat_params = {f"param_{k}": str(v) for k, v in (result.get("params") or {}).items()}
                            flat_params["architecture"] = str(result.get("architecture", "unknown"))
                            mlflow.log_params(flat_params)
                except Exception as e:
                    log(f"[MLFlow] Warning: failed to log trial metrics: {e}")
                # ------------------------------------
            else:
                log("[Real-Time Save WARNING] 'results_log' key missing in state. Data not saved to disk.")

    except subprocess.CalledProcessError as e:
        # Fallback for error reporting: Priority to e.stderr which now contains the full context
        worker_error = (e.stderr or "").strip() or (e.stdout or "").strip() or str(e) or "Unknown worker error"
        log(f"[WORKER ERROR] Training failed (Captured Output Available)")
        result = {"status": "error", "message": f"Training Worker Failure Details:\n{worker_error}"}
    except Exception as e:
        log(f"[ORCHESTRATION ERROR] {str(e)}")
        result = {"status": "error", "message": f"Orchestration error: {str(e)}"}
    finally:
        # Pulizia file temporaneo
        if os.path.exists(result_path):
            os.remove(result_path)

    return result


def _log_finetune_result(
    result: dict,
    thread_safe_state: dict,
    manifest_name: str,
    experiment_name: str,
) -> None:
    """
    Persist a completed finetune result in two places:
    1. thread_safe_state["results_log"] — flat entry compatible with run_training_trial output,
       so the analyst, _gather_cross_architecture_summary, and all existing readers can see it.
    2. PERSISTENT_PATHS["finetune_results_log"] — append-only JSON list with the full raw result,
       survives process restarts and is readable without a running agent.
    """
    import time as _time

    hp = result.get("hyperparameters", {})
    arch = (
        result.get("custom_architecture_file")
        or hp.get("representation_type")
        or result.get("representation_type")
        or experiment_name
    )

    flat_entry = {
        "status": "completed",
        "source": "finetune_trial",
        "run_id": result.get("run_id"),
        "timestamp": _time.time(),
        "architecture": arch,
        "manifest_name": manifest_name,
        "representation_type": result.get("representation_type") or hp.get("representation_type"),
        # Primary metrics — use finetune values as authoritative
        "held_out_test_accuracy": result.get("finetune", {}).get("ho_accuracy"),
        "shifted_test_accuracy": result.get("shifted_test_accuracy"),
        "distribution_shift_gap": result.get("distribution_shift_gap"),
        # Pretrain baseline for delta interpretation
        "pretrain_ho_accuracy": result.get("pretrain", {}).get("ho_accuracy"),
        "delta_pretrain_to_finetune": result.get("delta_pretrain_to_finetune"),
        "finetune_train_accuracy": result.get("finetune", {}).get("final_train_acc"),
        "finetune_unfreeze_last_n": result.get("finetune", {}).get("ft_unfreeze_last_n"),
        # Carry forward CV-style fields as None so readers don't crash
        "mean_accuracy": None,
        "mean_train_accuracy": None,
        "params": hp,
        "model_path": result.get("finetune", {}).get("model_path"),
        "seeds": result.get("seeds"),
        "per_class_finetune": result.get("finetune", {}).get("per_class"),
        "per_class_shifted": result.get("shifted_test_per_class"),
        # F1 / MCC — present only for runs after the 2026-05-28 upgrade; None for older entries
        "held_out_f1_macro":    result.get("finetune", {}).get("f1_macro"),
        "held_out_f1_weighted": result.get("finetune", {}).get("f1_weighted"),
        "held_out_mcc":         result.get("finetune", {}).get("mcc"),
        "shifted_f1_macro":     result.get("shifted_f1_macro"),
        "shifted_f1_weighted":  result.get("shifted_f1_weighted"),
        "shifted_mcc":          result.get("shifted_mcc"),
    }

    # 1. Append to in-memory log and save state
    if "results_log" in thread_safe_state:
        thread_safe_state["results_log"].append(flat_entry)
        try:
            save_state(thread_safe_state)
            log(f"[Finetune Log] Appended to results_log and saved state. "
                f"HO={flat_entry['held_out_test_accuracy']}, "
                f"Shifted={flat_entry['shifted_test_accuracy']}")
        except Exception as e:
            log(f"[Finetune Log] Warning: save_state failed: {e}")

    # 2. Append to finetune_results_log.json (full raw result + flat entry)
    try:
        ft_log_path = PERSISTENT_PATHS["finetune_results_log"]
        existing = []
        if os.path.exists(ft_log_path):
            with open(ft_log_path, "r") as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = []
        existing.append({"flat": flat_entry, "raw": result})
        with open(ft_log_path, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        log(f"[Finetune Log] Raw result written to {ft_log_path} ({len(existing)} total entries)")
    except Exception as e:
        log(f"[Finetune Log] Warning: finetune_results_log write failed: {e}")


# ---------------------------------------------------------------------------
# Finetune progress evaluator — reads finetune_results_log.json and produces a
# cross-architecture comparison so the Theorist (via Analyst A.6) and the Decider
# (just-in-time via tool call) can see which arch/representation combo is working,
# whether per-class collapse is happening, and where attention should shift.
# Single source of truth: _evaluate_finetune_progress_impl. Two consumers:
#   1) analyst._gather_finetune_summary  (auto, every cycle, formatted as markdown)
#   2) AVAILABLE_TOOLS["evaluate_finetune_progress"] (agent-callable, returns dict)
# ---------------------------------------------------------------------------

def _classify_finetune_diagnosis(n_trials, mean_shifted, std_class, chance_baseline, per_class_means):
    """Categorical diagnosis distinguishing failure modes that look similar on mean alone.
    The thresholds are intentionally hardcoded here (not config): they encode SEMANTIC
    decisions about what counts as collapse / random / learning, not tuning knobs."""
    if n_trials < 3:
        return "INSUFFICIENT_DATA"
    # Collapse detected if ANY class is effectively unpredicted (worker should still hit chance/3 randomly).
    if per_class_means:
        min_acc = min(per_class_means.values())
        if min_acc < 0.15:
            dominant = max(per_class_means.items(), key=lambda x: x[1])[0]
            return f"PARTIAL_COLLAPSE_FAVORING_{dominant}"
    delta = mean_shifted - chance_baseline
    if delta > 0.20 and std_class < 0.10:
        return "BALANCED_LEARNING"
    if delta > 0.20:
        return "STRONG_PERFORMANCE_WITH_BIAS"
    if abs(delta) < 0.05 and std_class < 0.10:
        return "RANDOM_UNIFORM_NEAR_CHANCE"
    return "MIXED_SUBOPTIMAL"


def _evaluate_finetune_progress_impl(last_n=None, cfg=None):
    """Read finetune_results_log.json and produce a cross-arch/rep comparison.

    Args:
        last_n: window size override. If None, read from config (training.finetune_progress_window),
                final fallback is 20 (with a visible log warning so a missing config key is not silent).
        cfg: agent config (dict or OmegaConf). Optional.

    Returns:
        dict with status, by_architecture (nested by representation), ranking_by_shifted_mean,
        and a descriptive direction string. Never raises on missing file — returns status='empty'.
    """
    # Resolve window size: explicit arg > config > hardcoded fallback (with warning)
    if last_n is None and cfg is not None:
        tcfg = cfg.get("training") if isinstance(cfg, dict) else getattr(cfg, "training", None)
        if tcfg is not None:
            _get = tcfg.get if isinstance(tcfg, dict) else (lambda k, d=None: getattr(tcfg, k, d))
            last_n = _get("finetune_progress_window", None)
    if last_n is None:
        last_n = 20
        log("[Finetune Progress] Warning: 'training.finetune_progress_window' not found in config, using default=20")
    last_n = int(last_n)

    ft_log_path = PERSISTENT_PATHS.get("finetune_results_log")
    if not ft_log_path or not os.path.exists(ft_log_path):
        return {
            "status": "empty",
            "window": f"last {last_n} trials",
            "n_trials_considered": 0,
            "direction": "No finetune trials have been recorded yet — run `run_training_trial_with_finetune` to begin domain adaptation.",
        }

    try:
        with open(ft_log_path, "r") as f:
            all_records = json.load(f)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load finetune_results_log.json: {e}"}

    if not all_records:
        return {"status": "empty", "n_trials_considered": 0,
                "direction": "finetune_results_log.json is empty."}

    # Keep the most recent N records (timestamp ordering; fall back to file order if missing).
    def _ts(r):
        return (r.get("flat", {}) or {}).get("timestamp", 0) or 0
    sorted_records = sorted(all_records, key=_ts)
    recent = sorted_records[-last_n:]
    n_considered = len(recent)

    # Bucket: architecture -> representation -> list[flat_entry]
    buckets = {}
    for r in recent:
        flat = r.get("flat", {}) or {}
        arch = flat.get("architecture") or "UNKNOWN_ARCH"
        rep = flat.get("representation_type") or "UNKNOWN_REP"
        buckets.setdefault(arch, {}).setdefault(rep, []).append(flat)

    # Derive chance baseline from any record that has per_class_shifted (n_classes = len(keys)).
    n_classes = None
    for r in recent:
        pcs = (r.get("flat", {}) or {}).get("per_class_shifted") or {}
        if pcs:
            n_classes = len(pcs)
            break
    chance_baseline = round(1.0 / n_classes, 3) if n_classes else 0.0

    by_architecture = {}
    ranking = []  # list of {arch, rep, shifted_mean}

    for arch, rep_buckets in buckets.items():
        n_total = sum(len(v) for v in rep_buckets.values())
        by_rep = {}
        best_rep = None
        best_rep_mean = -1.0

        for rep, trials in rep_buckets.items():
            shifted_vals = [t.get("shifted_test_accuracy") for t in trials if t.get("shifted_test_accuracy") is not None]
            if not shifted_vals:
                continue
            mean_shifted = float(np.mean(shifted_vals))
            max_shifted = float(np.max(shifted_vals))
            min_shifted = float(np.min(shifted_vals))
            std_shifted = float(np.std(shifted_vals)) if len(shifted_vals) > 1 else 0.0

            # Per-class means across this bucket's trials (skip trials missing the field).
            per_class_acc_lists = {}
            collapse_events = {}
            for t in trials:
                pcs = t.get("per_class_shifted") or {}
                for cls, d in pcs.items():
                    acc = d.get("acc") if isinstance(d, dict) else None
                    if acc is None:
                        continue
                    per_class_acc_lists.setdefault(cls, []).append(float(acc))
                    if acc < 0.05:
                        collapse_events[cls] = collapse_events.get(cls, 0) + 1
            per_class_mean = {c: round(float(np.mean(v)), 3) for c, v in per_class_acc_lists.items()}
            # Ensure every class shows up in collapse_events (default 0) so the LLM sees the full picture.
            for c in per_class_mean:
                collapse_events.setdefault(c, 0)

            # Variance across classes (within-trial dispersion), averaged across trials.
            per_trial_std = []
            for t in trials:
                pcs = t.get("per_class_shifted") or {}
                accs = [d.get("acc") for d in pcs.values() if isinstance(d, dict) and d.get("acc") is not None]
                if len(accs) > 1:
                    per_trial_std.append(float(np.std(accs)))
            avg_class_std = float(np.mean(per_trial_std)) if per_trial_std else 0.0

            diagnosis = _classify_finetune_diagnosis(
                n_trials=len(trials),
                mean_shifted=mean_shifted,
                std_class=avg_class_std,
                chance_baseline=chance_baseline,
                per_class_means=per_class_mean,
            )

            by_rep[rep] = {
                "n_trials": len(trials),
                "shifted_acc": {
                    "mean": round(mean_shifted, 3),
                    "max": round(max_shifted, 3),
                    "min": round(min_shifted, 3),
                    "std": round(std_shifted, 3),
                },
                "delta_to_chance": round(mean_shifted - chance_baseline, 3),
                "per_class_shifted_mean": per_class_mean,
                "per_class_std_across_classes": round(avg_class_std, 3),
                "collapse_events": collapse_events,
                "diagnosis": diagnosis,
            }

            if mean_shifted > best_rep_mean:
                best_rep_mean = mean_shifted
                best_rep = rep

            ranking.append({"arch": arch, "rep": rep, "shifted_mean": round(mean_shifted, 3),
                            "diagnosis": diagnosis})

        by_architecture[arch] = {
            "n_trials_total": n_total,
            "best_representation": best_rep,
            "by_representation": by_rep,
        }

    ranking.sort(key=lambda x: x["shifted_mean"], reverse=True)

    # Descriptive direction — no prescriptive params, just summarize the comparative state.
    lines = []
    if ranking:
        top = ranking[0]
        lines.append(f"Best: {top['arch']}/{top['rep']} shifted={top['shifted_mean']} ({top['diagnosis']}).")
    for r in ranking[1:]:
        lines.append(f"Other: {r['arch']}/{r['rep']} shifted={r['shifted_mean']} ({r['diagnosis']}).")
    # Highlight chronic per-class collapse across ALL buckets.
    chronic = {}
    total_buckets = 0
    for arch_data in by_architecture.values():
        for rep_data in arch_data["by_representation"].values():
            total_buckets += 1
            for cls, count in rep_data["collapse_events"].items():
                if count > 0:
                    chronic[cls] = chronic.get(cls, 0) + count
    if chronic and total_buckets:
        worst_cls = max(chronic.items(), key=lambda x: x[1])
        lines.append(f"Recurring class collapse (<5% acc): {worst_cls[0]} in {worst_cls[1]} trial(s) across all buckets.")
    direction = " ".join(lines) if lines else "No actionable signal yet."

    return {
        "status": "ok",
        "window": f"last {last_n} trials",
        "n_trials_considered": n_considered,
        "n_classes": n_classes,
        "chance_baseline": chance_baseline,
        "by_architecture": by_architecture,
        "ranking_by_shifted_mean": ranking,
        "direction": direction,
    }


def evaluate_finetune_progress(thread_safe_state, last_n=None):
    """Agent-callable wrapper. Reads cfg from state so the Decider/Theorist can call
    it without passing config explicitly. The result dict is returned verbatim to the
    agent; the Analyst formats the same dict as markdown for Part 1 section A.6."""
    cfg = thread_safe_state.get("cfg") if isinstance(thread_safe_state, dict) else getattr(thread_safe_state, "cfg", None)
    return _evaluate_finetune_progress_impl(last_n=last_n, cfg=cfg)


def run_training_trial_with_finetune(
    thread_safe_state: Dict[str, Any],
    manifest_name: str,
    representation_type: Optional[str] = None,
    experiment_name: Optional[str] = None,
    custom_architecture_file: Optional[str] = None,
    seeds: Optional[list] = None,
    **hyperparameters
) -> Dict[str, Any]:
    """
    ESCALATION TOOL — Two-phase pretrain + full-model fine-tune on a committed
    architecture. Always saves the final model to disk.

    Use ONLY when:
      1. You have identified an architecture/config as your best candidate.
      2. Standard `run_training_trial` HO has plateaued (3+ trials without
         improvement on the current best).
      3. The manifest has a finetune split available (`has_finetune_split: true`
         in datasets_metadata.json).

    This tool is expensive (multi-seed pretrain + full-model finetune).
    For exploration, keep using `run_training_trial` instead.

    Returns rich result with:
      - pretrain.ho_accuracy  (best seed)
      - finetune.ho_accuracy  (final committed model)
      - delta_pretrain_to_finetune
      - model_path on disk
    """
    # GPU id handling (same as run_training_trial)
    gpu_id = hyperparameters.pop("gpu_id", None)

    # --- SESSION CAP (from config.yaml training.finetune_session_cap) ---
    cfg = thread_safe_state.get("cfg")
    _training_cfg = (cfg.get("training") if isinstance(cfg, dict) else getattr(cfg, "training", None)) if cfg else None
    _get_cfg = (lambda k, d: _training_cfg.get(k, d)) if isinstance(_training_cfg, dict) else (lambda k, d: getattr(_training_cfg, k, d)) if _training_cfg else (lambda k, d: d)
    _finetune_cap = _get_cfg("finetune_session_cap", 2)
    if _finetune_cap > 0:
        _finetune_calls = thread_safe_state.get("finetune_calls_this_session", 0)
        if _finetune_calls >= _finetune_cap:
            return {
                "status": "error",
                "message": (
                    f"[USER_GATE] run_training_trial_with_finetune has been called {_finetune_calls} time(s) "
                    f"this session (cap={_finetune_cap}, set in config.yaml training.finetune_session_cap). "
                    f"STOP and surface this to the user. Only continue if the user explicitly says "
                    f"'run finetune again' or 'unlock finetune'. Do NOT call this tool again autonomously."
                )
            }
        thread_safe_state["finetune_calls_this_session"] = _finetune_calls + 1

    # --- REPRESENTATION TYPE VALIDATION (before triggering any generation) ---
    _VALID_REP_TYPES = [
        "1D_CNN", "2D_SPECTROGRAM", "2D_GAF", "2D_CWT_SCALOGRAM",
        "2D_GENERIC_IMAGE", "2D_IMAGE", "3D_VIDEO", "3D_GAF_VIDEO",
        "3D_DYNAMIC_GAF", "3D_DYNAMIC_CWT", "3D_WAVELET_CWT"
    ]
    if representation_type and representation_type not in _VALID_REP_TYPES:
        return {
            "status": "error",
            "message": (
                f"Invalid representation_type '{representation_type}'. "
                f"Valid types: {_VALID_REP_TYPES}. "
                f"Common mistake: '3D_GAF' does not exist — use '3D_DYNAMIC_GAF' or '3D_GAF_VIDEO'."
            )
        }

    log(f"[Finetune Tool] Manifest='{manifest_name}'  Rep='{representation_type}'  "
        f"Arch='{custom_architecture_file or experiment_name}'  GPU={gpu_id}  "
        f"(session call {thread_safe_state.get('finetune_calls_this_session', 1)}/{_finetune_cap if _finetune_cap > 0 else 'unlimited'})")

    # Validate manifest + finetune availability
    try:
        metadata = _load_dataset_metadata(thread_safe_state)
    except Exception as e:
        return {"status": "error", "message": f"[SYSTEM_ERROR] Failed to load metadata: {e}"}

    if manifest_name not in metadata:
        return {
            "status": "error",
            "message": f"Manifest '{manifest_name}' not found. Run create_dataset_manifest first."
        }

    manifest_data = metadata[manifest_name]

    # Resolve representation early (needed for generation checks below)
    effective_representation = representation_type
    if not effective_representation:
        avail = list(manifest_data.get("generated_representations", {}).keys())
        effective_representation = avail[0] if avail else "1D_CNN"
        log(f"[Finetune Tool] Auto-selected representation: '{effective_representation}'")

    # Auto-generate data if training rep or finetune split is missing.
    # The data pipeline auto-discovers finetune_data_folder/ and shifted_test_folder/
    # in the same run that generates the training data, so one call is enough.
    needs_gen = False
    if effective_representation not in manifest_data.get("generated_representations", {}):
        needs_gen = True
        log(f"[Finetune Tool] Training data missing for '{effective_representation}'. Triggering generation...")
    elif not manifest_data.get("has_finetune_split"):
        needs_gen = True
        log(f"[Finetune Tool] Finetune split not yet in metadata. Re-running data pipeline...")
    elif effective_representation not in manifest_data.get("generated_finetune_representations", {}):
        needs_gen = True
        log(f"[Finetune Tool] Finetune split missing for rep '{effective_representation}'. Re-running...")

    if needs_gen:
        gen_result = generate_representation(manifest_name, effective_representation, thread_safe_state)
        if gen_result.get("status") != "completed":
            detailed = gen_result.get("message", "Unknown error")
            if gen_result.get("worker_error"):
                detailed += f" | Details: {gen_result.get('worker_error')}"
            return {"status": "error", "message": f"[SYSTEM_ERROR] Auto-generation failed: {detailed}"}
        metadata = _load_dataset_metadata(thread_safe_state)
        manifest_data = metadata[manifest_name]
        log(f"[Finetune Tool] Data generated. Proceeding to finetune validation...")

    # Confirm finetune split exists after potential generation
    if not manifest_data.get("has_finetune_split"):
        return {
            "status": "error",
            "message": (
                f"Manifest '{manifest_name}' has no finetune split. "
                f"This tool requires X_finetune.npy + y_finetune.npy in the raw data source "
                f"(or in finetune_data_folder/ or test_data_folder_clinical/). "
                f"Use run_training_trial instead, OR add finetune data and retry."
            )
        }

    ft_avail = manifest_data.get("generated_finetune_representations", {})
    if effective_representation not in ft_avail:
        return {
            "status": "error",
            "message": (
                f"Finetune data generated but not for rep '{effective_representation}'. "
                f"Available finetune reps: {list(ft_avail.keys())}. "
                f"Re-run generate_representation for '{effective_representation}'."
            )
        }

    # Pre-flight: custom architecture file
    if custom_architecture_file:
        arch_path = os.path.join(PERSISTENT_PATHS["custom_architectures_dir"], custom_architecture_file)
        if not os.path.exists(arch_path) and not os.path.exists(arch_path + ".py"):
            return {
                "status": "error",
                "message": f"Custom architecture file '{custom_architecture_file}' not found."
            }

    # Inject representation and config defaults
    hyperparameters["representation_type"] = effective_representation
    if "epochs" not in hyperparameters:
        hyperparameters["epochs"] = 150
    cfg = thread_safe_state.get("cfg")
    if cfg:
        training_cfg = cfg.get("training") if isinstance(cfg, dict) else getattr(cfg, "training", None)
        if training_cfg:
            _get = (lambda k, d: training_cfg.get(k, d)) if isinstance(training_cfg, dict) else (lambda k, d: getattr(training_cfg, k, d))
            if "patience" not in hyperparameters:
                hyperparameters["patience"] = _get("early_stopping_patience", 20)

    data_path_root = os.path.join(PERSISTENT_PATHS["generated_datasets_dir"], manifest_name)

    # Dispatch Celery task
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json',
                                          dir=PERSISTENT_PATHS["processed_data_dir"]) as tmp:
            result_path = tmp.name

        log(f"[Finetune Tool] Dispatching task (Exp: {experiment_name}, seeds={seeds or '[42,1,7]'})...")
        task_result = run_finetune_training_worker.apply_async(
            args=[experiment_name or "finetune_run", hyperparameters, result_path],
            kwargs={
                "data_path_root": data_path_root,
                "custom_architecture_file": custom_architecture_file,
                "seeds": seeds,
                "gpu_id": gpu_id,
            }
        )
        log(f"[Finetune Tool] Task dispatched: ID={task_result.id}")

        start_time = time.time()
        max_wait_seconds = 3600 * 48
        try:
            process_output = task_result.get(timeout=max_wait_seconds)
        except Exception as e:
            elapsed_mins = int((time.time() - start_time) / 60)
            log(f"[Finetune Tool] FAILED/TIMEOUT after {elapsed_mins} min: {e}")
            try:
                task_result.revoke(terminate=True)
            except Exception:
                pass
            return {"status": "error", "message": f"Celery task failed: {e}"}

        duration_min = int((time.time() - start_time) / 60)
        log(f"[Finetune Tool] Completed in {duration_min} min")

        # --- PERSIST FINETUNE RESULT ---
        if isinstance(process_output, dict) and process_output.get("status") == "completed":
            _log_finetune_result(process_output, thread_safe_state, manifest_name, experiment_name)

        return process_output

    except Exception as e:
        return {"status": "error", "message": f"Orchestration failed: {e}\n{traceback.format_exc()}"}
    finally:
        if "result_path" in locals() and os.path.exists(result_path):
            try:
                os.remove(result_path)
            except Exception:
                pass


def run_imagenet_optimization_trial(
    thread_safe_state: Dict[str, Any],
    custom_architecture_file: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Runs a fast optimization trial on a subset of the ImageNet dataset using a dynamically 
    generated architecture. Designed specifically for the AI Agent to perform Auto-ML on ILSVRC.

    Args:
        thread_safe_state: Shared state to log results.
        custom_architecture_file: Python file containing `build_model(input_shape, num_classes, **kwargs)`.
        kwargs: Hyperparameters to pass to the model and training loop (e.g., subset_fraction, learning_rate).
    """
    log(f"Starting ImageNet optimization trial with architecture: {custom_architecture_file}")
    
    # Pre-flight check file existence to prevent cluster freeze
    if custom_architecture_file:
        __arch_path = os.path.join(PERSISTENT_PATHS["custom_architectures_dir"], custom_architecture_file)
        if not os.path.exists(__arch_path) and not os.path.exists(__arch_path + ".py"):
            return {
                "status": "error",
                "message": f"[ERRORE AGENTE]: Il file '{custom_architecture_file}' non e' stato trovato in '{PERSISTENT_PATHS['custom_architectures_dir']}'. Hai dimenticato di crearlo con 'write_architecture_file'?"
            }

    # 1. Resolve Data Paths Dynamically from Config
    cfg = thread_safe_state.get("cfg", {})
    imagenet_cfg = cfg.get("imagenet", {}) if hasattr(cfg, "get") else {}
    
    # Base fallback if not in config
    raw_root = PERSISTENT_PATHS.get("raw_data_dir", "processed_data/raw_data")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    train_dir_rel = imagenet_cfg.get("dataset_dir", "processed_data/imagenet/train")
    val_dir_rel = imagenet_cfg.get("val_dir", "processed_data/imagenet/val")
    annotation_dir_rel = imagenet_cfg.get("annotation_dir", None)
    synset_map_rel = imagenet_cfg.get("synset_map", "imagenet/synset_mapping.txt")

    # If the relative path doesn't start with raw_data/processed_data, prepend project root correctly
    # The config usually specifies "processed_data/..." so joining with project_root is sufficient.
    train_dir = train_dir_rel if os.path.isabs(train_dir_rel) else os.path.normpath(os.path.join(project_root, train_dir_rel))
    val_dir = val_dir_rel if os.path.isabs(val_dir_rel) else os.path.normpath(os.path.join(project_root, val_dir_rel))
    
    annotation_dir = None
    if annotation_dir_rel:
         annotation_dir = annotation_dir_rel if os.path.isabs(annotation_dir_rel) else os.path.normpath(os.path.join(project_root, annotation_dir_rel))

    synset_map = synset_map_rel if os.path.isabs(synset_map_rel) else os.path.normpath(os.path.join(project_root, synset_map_rel))

    if not os.path.exists(train_dir):
         # Try looking in raw_data if processed_data/imagenet/train fails
         alt_train_dir = os.path.normpath(os.path.join(project_root, "processed_data/raw_data/ILSVRC/Data/CLS-LOC/train"))
         alt_val_dir = os.path.normpath(os.path.join(project_root, "processed_data/raw_data/ILSVRC/Data/CLS-LOC/val"))
         if os.path.exists(alt_train_dir):
             train_dir = alt_train_dir
             val_dir = alt_val_dir
             synset_map = os.path.normpath(os.path.join(project_root, "processed_data/raw_data/ILSVRC/LOC_synset_mapping.txt"))
             annotation_dir = os.path.normpath(os.path.join(project_root, "processed_data/raw_data/ILSVRC/Annotations/CLS-LOC"))
         else:
             # DRY-RUN FALLBACK FOR TESTING/AGENT VERIFICATION (Bypasses the 150GB requirement locally)
             log("[WARNING] ImageNet dataset not found. Falling back to DRY-RUN mode for agent verification.")
             return {
                 "status": "completed", 
                 "mean_accuracy": 0.05, 
                 "inference_time_ms": 12.5, 
                 "params": 500000, 
                 "message": "DRY RUN COMPLETE: Dataset missing locally. Simulating successful architecture compilation.",
                 "hyperparameters": kwargs,
                 "model_path": None
             }

    # Extract optimization-specific kwargs
    subset_fraction = kwargs.pop("subset_fraction", 0.1) # Default 10%
    epochs = kwargs.pop("epochs", 5)                     # Fast iterations
    batch_size = kwargs.pop("batch_size", 128)
    learning_rate = kwargs.pop("learning_rate", 0.001)

    opt_args = {
        "dataset_dir": train_dir,
        "val_dir": val_dir,
        "annotation_dir": annotation_dir,
        "synset_map": synset_map,
        "custom_architecture_file": custom_architecture_file,
        "hyperparameters": {
            "subset_fraction": subset_fraction,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            **kwargs # Pass along agent's architectural hyperparams
        }
    }

    try:
        # Dispatch to Celery
        log(f"Dispatching ImageNet optimization to Celery (Subset: {subset_fraction*100}%, Epochs: {epochs})")
        async_result = run_imagenet_optimization_task.apply_async(args=[opt_args])
        result = async_result.get(timeout=86400) # 24h max
        
        # Enforce agent analytical compatibility format
        if result.get("status") == "completed":
             result["architecture"] = custom_architecture_file
             result["manifest_name"] = "ILSVRC_ImageNet"
             result["source"] = "agent_trial"
             
             # Log to central state for analysis tools (get_correlation_matrix, etc.)
             if "results_log" in thread_safe_state:
                 thread_safe_state["results_log"].append(result)
                 save_state(thread_safe_state)
                 log(f"[Real-Time Save] ImageNet trial logged for {custom_architecture_file}.")
        else:
             log(f"ImageNet trial failed: {result.get('message', 'Unknown error')}")

        return result
        
    except Exception as e:
        log(f"Error during ImageNet optimization trial: {e}")
        return {"status": "error", "message": f"Orchestration error: {e}"}

def get_recent_trials(thread_safe_state: Dict[str, Any], n: int = 5, architecture_filter: Optional[str] = None) -> Dict[str, Any]:
    """Shows the raw results of the last 'n' trials, not sorted by accuracy. Can be filtered by architecture."""
    log(f"Retrieving the last {n} trials... (Filter: {architecture_filter})")
    
    df_flat = _get_and_flatten_log(architecture_filter, thread_safe_state)

    if df_flat is None:
        return {"status": "error", "message": "The results log is empty or no matching trials."}

    if architecture_filter and "architecture" in df_flat.columns:
        df_flat = df_flat[df_flat["architecture"] == architecture_filter].copy()
        if df_flat.empty:
            return {
                "status": "completed",
                "recent_trials": [],
                "message": f"No trials found for architecture '{architecture_filter}'.",
            }

    return {"status": "completed", "recent_trials": df_flat.tail(n).to_dict("records")}


def get_correlation_matrix(thread_safe_state: Dict[str, Any], architecture_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes the results history to show how each hyperparameter correlates with accuracy and stability.
    Now includes P-Values (if scipy available) to filter out statistical noise (p > 0.05).
    """
    log(f"Calculating correlation matrix... (Filter: {architecture_filter})")

    df_flat = _get_and_flatten_log(architecture_filter, thread_safe_state)

    if df_flat is None or len(df_flat) < 5: # Need more samples for p-values
        return {"status": "error", "message": "Not enough data for correlation analysis (need > 5 samples)."}

    df_numeric = df_flat.select_dtypes(include=np.number)

    if "mean_accuracy" not in df_numeric.columns:
        return {"status": "error", "message": "'mean_accuracy' not found."}

    results = {}
    
    # Target columns to correlate against
    targets = ["mean_accuracy"]
    if "std_accuracy" in df_numeric.columns:
        targets.append("std_accuracy")

    for target in targets:
        target_correlations = {}
        for col in df_numeric.columns:
            if col == target: 
                continue
                
            # Basic Correlation
            coeff = df_numeric[col].corr(df_numeric[target])
            if pd.isna(coeff):
                continue

            stats_info = {"correlation": round(coeff, 3)}

            # Statistical Significance (P-Value)
            if SCIPY_AVAILABLE:
                try:
                    # Drop NaNs for pair-wise calculation
                    valid_data = df_numeric[[col, target]].dropna()
                    if len(valid_data) > 2:
                        p_value = float(stats.pearsonr(valid_data[col], valid_data[target])[1])
                        stats_info["p_value"] = round(p_value, 4)
                        stats_info["significant"] = bool(p_value < 0.05)
                    else:
                         stats_info["p_value"] = "N/A (too few samples)"
                except Exception:
                    stats_info["p_value"] = "Error"
            else:
                stats_info["p_value"] = "Not Installed (scipy)"
            
            target_correlations[col] = stats_info

        # Sort by absolute correlation strength (cast correlation float explicitly)
        sorted_corrs = dict(sorted(target_correlations.items(), key=lambda item: float(abs(item[1]["correlation"])), reverse=True))
        results[target] = sorted_corrs

    return {"status": "completed", "correlation_analysis": results}


def get_parameter_importance(thread_safe_state: Dict[str, Any], architecture_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Determines the impact of each hyperparameter on model accuracy using a Random Forest model.

    This tool trains a RandomForestRegressor on the historical trial data to evaluate
    which hyperparameters are the most influential in predicting the 'mean_accuracy'.
    It returns a sorted list of parameters by their importance score. Can be filtered by architecture.
    """
    log(f"Calculating parameter importance... (Filter: {architecture_filter})")

    df_flat = _get_and_flatten_log(architecture_filter, thread_safe_state)

    if df_flat is None or len(df_flat) < 5:
        return {"status": "error", "message": "At least 5 results are needed for importance analysis."}

    # Drop rows where target variable 'mean_accuracy' is NaN
    df_flat = df_flat.dropna(subset=["mean_accuracy"])
    
    if len(df_flat) < 5:
         return {"status": "error", "message": "Not enough valid trials (with accuracy) for analysis."}

    y = df_flat["mean_accuracy"]
    X = df_flat.drop(columns=list(METADATA_COLUMNS), errors="ignore").select_dtypes(include=np.number)
    X = X.fillna(X.median())

    if X.empty:
        return {"status": "error", "message": "No numerical hyperparameters found to analyze."}

    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    return {"status": "completed", "importance": json.loads(importance.to_json())}


def _get_mode_or_median(series: pd.Series) ->  Union[int, float]:
    """
    Returns the Mode (most frequent) for integer-like data to avoid fractional values
    (e.g. Kernel Size 3.67). Falls back to Median if mode is ambiguous or data is float.
    """
    try:
        # Check if it's conceptually an integer (even if float typed)
        is_integer_col = False
        if pd.api.types.is_integer_dtype(series):
            is_integer_col = True
        elif pd.api.types.is_float_dtype(series):
            # Check if all non-NaN values are effectively integers (e.g. 3.0, 4.0)
            if series.dropna().apply(lambda x: x.is_integer()).all():
                is_integer_col = True

        if is_integer_col:
            # Return Mode (most common value)
            modes = series.mode()
            if not modes.empty:
                return int(modes[0]) 
            # Fallback to median if no mode (rare)
            return int(series.median())
        
        return series.mean()
    except Exception:
        return series.mean()


def analyze_best_vs_worst_trials(thread_safe_state: Dict[str, Any], architecture_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Compares the average hyperparameter values of the top-performing trials against the bottom-performing ones.
    UPDATED: Uses Mode/Median for discrete parameters (kernels, filters) to avoid "3.67" values.
    UPDATED: Checks for Overfitting (Train vs Test gap).
    """
    log(f"Analyzing best vs. worst trials... (Filter: {architecture_filter})")

    df_flat = _get_and_flatten_log(architecture_filter, thread_safe_state)

    if df_flat is None or len(df_flat) < 10:
        return {"status": "error", "message": "At least 10 results are required."}

    df_sorted = df_flat.sort_values(by="mean_accuracy", ascending=False)
    n = max(1, int(len(df_sorted) * 0.25))
    
    numeric_cols = df_sorted.select_dtypes(include=np.number).columns
    
    # Helper to aggregate stats specifically tailored for the column type
    def _smart_aggregate(dataframe_chunk):
        stats = {}
        for col in numeric_cols:
            if col in ["num_filters", "kernel_size", "num_layers", "epochs", "batch_size"]:
                # Discrete -> Mode/Median
                stats[col] = _get_mode_or_median(dataframe_chunk[col])
            else:
                # Continuous -> Mean
                stats[col] = dataframe_chunk[col].mean()
        return stats

    best_stats = _smart_aggregate(df_sorted.head(n))
    worst_stats = _smart_aggregate(df_sorted.tail(n))

    # --- Single Best and Worst Trial (concrete records, not averages) ---
    hyperparam_cols = get_hyperparameter_columns(df_sorted)
    def _row_to_dict(row):
        d = {}
        for col in hyperparam_cols:
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                d[col] = val
        return d

    single_best = _row_to_dict(df_sorted.iloc[0])
    single_worst = _row_to_dict(df_sorted.iloc[-1])

    # --- Check for Overfitting (Generalization Gap) ---
    overfitting_analysis = "No train accuracy data available."
    if "mean_train_accuracy" in best_stats and "mean_accuracy" in best_stats: # mean_accuracy is usually test/val
        train_acc = best_stats["mean_train_accuracy"]
        test_acc = best_stats["mean_accuracy"]
        gap = train_acc - test_acc

        if gap > 0.05: # > 5% gap
            overfitting_analysis = f"WARNING: High Overfitting Risk. Train Acc ({train_acc:.3f}) is significantly higher than Test Acc ({test_acc:.3f}). Gap: {gap:.1%}"
        elif gap < -0.02:
            overfitting_analysis = f"Note: Test accuracy is higher than Train ({gap:.1%}). Likely underfitting or strong regularization."
        else:
            overfitting_analysis = f"Healthy Generalization. Gap: {gap:.1%}"

    return {
        "status": "completed",
        "note": f"Stats are averages over top/bottom {n} of {len(df_sorted)} total trials. Use 'single_best_trial' for the exact best config.",
        "single_best_trial": single_best,
        "single_worst_trial": single_worst,
        "best_trials_stats": best_stats,
        "worst_trials_stats": worst_stats,
        "overfitting_check": overfitting_analysis
    }


def analyze_hyperparameter_tradeoffs(thread_safe_state: Dict[str, Any], architecture_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes hyperparameter trade-offs for accuracy and stability. Can be filtered by architecture.
    """
    log(f"Analyzing hyperparameter trade-offs... (Filter: {architecture_filter})")

    df_flat = _get_and_flatten_log(architecture_filter, thread_safe_state)

    if df_flat is None or len(df_flat) < 10:
        return {"status": "error", "message": "At least 10 results are needed for this analysis."}

    hyperparameters = [
        col for col in get_hyperparameter_columns(df_flat)
        if pd.api.types.is_numeric_dtype(df_flat[col])
    ]

    tradeoff_report = {}

    for param in hyperparameters:
        analysis = {}
        if df_flat[param].nunique() <= 5:
            analysis["type"] = "categorical"
            grouped = (
                df_flat.groupby(param)
                .agg(
                    mean_accuracy=("mean_accuracy", "mean"),
                    std_accuracy=("std_accuracy", "mean"),
                    trial_count=("mean_accuracy", "size"),
                )
                .reset_index()
            )
            analysis["analysis"] = json.loads(grouped.to_json(orient="records"))
        else:
            try:
                analysis["type"] = "numerical"
                df_flat["quantile"] = pd.qcut(df_flat[param], 3, duplicates="drop")
                grouped = (
                    df_flat.groupby("quantile")
                    .agg(
                        mean_accuracy=("mean_accuracy", "mean"),
                        std_accuracy=("std_accuracy", "mean"),
                        trial_count=("mean_accuracy", "size"),
                    )
                    .reset_index()
                )
                analysis["analysis"] = json.loads(grouped.to_json(orient="records"))
            except Exception as e:
                log(f"[TOOL WARNING] Could not analyze parameter '{param}' as numerical: {e}")
                continue
        tradeoff_report[param] = analysis

    return {"status": "completed", "tradeoffs": tradeoff_report}


def get_explored_ranges(
    thread_safe_state: Dict[str, Any],
    architecture_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Reports which hyperparameters have been explored vs never varied in past trials.
    Parameters that are always constant (or never passed) are silent defaults — invisible
    confounders that must be varied in the next trial.
    """
    log(f"Computing explored ranges... (Filter: {architecture_filter})")

    df = _get_and_flatten_log(architecture_filter, thread_safe_state)
    if df is None or len(df) < 3:
        return {"status": "error", "message": "At least 3 trials are required for coverage analysis."}

    REQUIRED_PARAMS = [
        "learning_rate", "dropout_rate", "filters", "kernel_size",
        "batch_size", "epochs", "patience", "weight_decay", "batch_norm"
    ]

    report = {}
    never_varied = []

    for param in REQUIRED_PARAMS:
        if param not in df.columns:
            report[param] = {
                "status": "never_passed",
                "note": "always used worker silent default — value is INVISIBLE in logs"
            }
            never_varied.append(param)
            continue

        col = df[param].dropna()
        if len(col) == 0:
            report[param] = {"status": "never_passed", "note": "column present but all NaN"}
            never_varied.append(param)
        elif col.nunique() == 1:
            report[param] = {
                "status": "never_varied",
                "only_value": float(col.iloc[0]) if pd.api.types.is_numeric_dtype(col) else col.iloc[0],
                "n_trials": int(len(col)),
                "note": "always constant — likely a silent default, explore other values"
            }
            never_varied.append(param)
        elif pd.api.types.is_numeric_dtype(col):
            vals = sorted(col.dropna().unique().tolist())
            report[param] = {
                "status": "explored",
                "min_tested": float(col.min()),
                "max_tested": float(col.max()),
                "n_unique_values": int(col.nunique()),
                "n_trials": int(len(col)),
                "values_tested": [float(v) for v in vals[:10]]
            }
        else:
            report[param] = {
                "status": "explored",
                "values_tested": col.unique().tolist(),
                "n_trials": int(len(col))
            }

    action = (
        f"⚠️ PRIORITY: these params were never varied → explore them next: {never_varied}"
        if never_varied else "✅ Good coverage — all required params have been varied."
    )

    return {
        "status": "completed",
        "architecture_filter": architecture_filter,
        "n_trials_analyzed": int(len(df)),
        "explored_ranges": report,
        "never_varied_params": never_varied,
        "action": action
    }


def analyze_hyperparameter_trend(
    param_x: str,
    param_y: str,
    thread_safe_state: Dict[str, Any],
    architecture_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyzes the trend between two parameters (e.g., hyperparameter vs metric).
    Calculates Linear Regression slope for numerical X, or group means for categorical X.
    
    Args:
        param_x: The independent variable (e.g., 'learning_rate', 'num_layers').
        param_y: The dependent variable (usually 'mean_accuracy').
        thread_safe_state: System state containing results_log.
        architecture_filter: Optional filter for a specific model architecture.
    """
    log(f"Analyzing trend: {param_x} vs {param_y} (Filter: {architecture_filter})")

    df_flat = _get_and_flatten_log(architecture_filter, thread_safe_state)

    if df_flat is None or len(df_flat) < 3:
        return {"status": "error", "message": "At least 3 valid trials are required for trend analysis."}

    if param_x not in df_flat.columns or param_y not in df_flat.columns:
        return {
            "status": "error", 
            "message": f"Parameters not found in logs. Available: {list(df_flat.columns)}"
        }

    # Data Cleaning: Drop NaNs for the specific columns
    df_clean = df_flat.dropna(subset=[param_x, param_y])
    if len(df_clean) < 3:
         return {"status": "error", "message": f"Not enough valid data points after dropping NaNs for {param_x}/{param_y}."}

    is_x_numeric = pd.api.types.is_numeric_dtype(df_clean[param_x])
    is_y_numeric = pd.api.types.is_numeric_dtype(df_clean[param_y])

    result = {
        "status": "completed",
        "param_x": param_x,
        "param_y": param_y,
        "n_samples": len(df_clean),
        "architecture": architecture_filter or "Global"
    }

    if is_x_numeric and is_y_numeric:
        # NUMERICAL TREND (Linear Regression)
        X = df_clean[param_x].values.reshape(-1, 1)
        y = df_clean[param_y].values
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            slope = float(model.coef_[0])
            intercept = float(model.intercept_)
            r_sq = float(model.score(X, y))
            correlation = float(df_clean[[param_x, param_y]].corr().iloc[0, 1])

            result["analysis_type"] = "linear_regression"
            result["slope"] = slope
            result["intercept"] = intercept
            result["r_squared"] = r_sq
            result["correlation"] = correlation
            result["interpretation"] = (
                f"For every 1 unit increase in '{param_x}', '{param_y}' changes by approx {slope:.4f}. "
                f"Correlation: {correlation:.2f}."
            )
        except Exception as e:
            result["error"] = f"Regression failed: {e}"
    
    # GROUP ANALYSIS (Useful for both categorical and discrete numeric X)
    try:
        # If numeric but few unique values, treat as discrete categories for better insight
        unique_x = df_clean[param_x].nunique()
        
        # Aggregate stats
        grouped = df_clean.groupby(param_x)[param_y].agg(['mean', 'std', 'count']).reset_index()
        grouped = grouped.sort_values(by=param_x)
        
        result["group_analysis"] = json.loads(grouped.to_json(orient="records"))
        
        if not is_x_numeric or unique_x <= 10:
             # Identify best group
             best_group = grouped.sort_values(by="mean", ascending=False).iloc[0]
             result["best_setting"] = {
                 param_x: best_group[param_x],
                 f"avg_{param_y}": float(best_group["mean"]),
                 "sample_count": int(best_group["count"])
             }
    except Exception as e:
        log(f"[Tool Warning] Group analysis failed in trend tool: {e}")

    return result


def get_sorted_trials(
    thread_safe_state: Dict[str, Any],
    sort_by: str = "mean_accuracy",
    ascending: bool = False,
    n: int = 10,
    architecture_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Displays top trials sorted by a metric. Can be filtered by architecture.
    """
    log(f"Retrieving top {n} trials sorted by '{sort_by}'... (Filter: {architecture_filter})")

    df_flat = _get_and_flatten_log(architecture_filter, thread_safe_state)

    if df_flat is None:
        return {"status": "error", "message": "The results log is empty."}

    if sort_by not in df_flat.columns:
        valid_options = [col for col in df_flat.columns if pd.api.types.is_numeric_dtype(df_flat[col])]
        return {"status": "error", "message": f"Metric '{sort_by}' not found. Valid options: {valid_options}"}


    try:
        sorted_df = df_flat.sort_values(by=sort_by, ascending=ascending).head(n)
        
        count = len(sorted_df)
        log(f"[Tool: get_sorted_trials] Found {count} trials sorted by {sort_by}.")

        result_json = json.loads(sorted_df.to_json(orient="records"))
        return {"status": "completed", f"top_{n}_trials_by_{sort_by}": result_json}
    except Exception as e:
        error_msg = f"Error during data sorting: {e}"
        log(f"[TOOL ERROR] {error_msg}")
        return {"status": "error", "message": error_msg}



def analyze_strategy_effectiveness(thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analizza la cronologia strategica per valutare quali strumenti e tipi di ipotesi
    hanno portato ai maggiori miglioramenti di accuratezza.
    """
    log("Analyzing strategic effectiveness...")
    strategic_log = thread_safe_state.get("strategic_log", [])

    if len(strategic_log) < 3:
        return {
            "status": "error",
            "message": "Dati strategici insufficienti per un'analisi (richiesti almeno 3 punti dati).",
        }

    df = pd.DataFrame(strategic_log)

    tool_effectiveness = df.groupby("tool_name")["improvement"].mean().sort_values(ascending=False).to_dict()

    df_sorted_by_improvement = df.sort_values(by="improvement", ascending=False)

    top_hypotheses = df_sorted_by_improvement.head(3)[["hypothesis", "improvement"]].to_dict("records")
    worst_hypotheses = df_sorted_by_improvement.tail(3)[["hypothesis", "improvement"]].to_dict("records")

    report = {
        "tool_vs_improvement_avg": tool_effectiveness,
        "top_3_successful_hypotheses": top_hypotheses,
        "bottom_3_successful_hypotheses": worst_hypotheses,
    }

    return {"status": "completed", "effectiveness_report": report}


def summarize_findings(thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reflects on the entire experiment history to extract key insights.
    This tool synthesizes data from long-term memory, best trials, and parameter correlations
    to provide a comprehensive summary, ignoring the volatile short-term chat history.
    """
    log("Synthesizing a comprehensive summary from all available data...")

    try:
        log("[Summary] Recalling key memories...")
        recalled_memories = recall_relevant_memories(
            query="Key successful strategies, architectural insights, and important hyperparameter findings",
            thread_safe_state=thread_safe_state
        )
        if recalled_memories.get("status") == "completed":
            memory_summary = recalled_memories.get("memories", "No relevant memories found.")
            if isinstance(memory_summary, list):
                memory_summary = "\n- ".join(memory_summary)
        else:
            memory_summary = "Could not access long-term memory."
    except Exception as e:
        memory_summary = f"Error recalling memories: {e}"

    try:
        log("[Summary] Analyzing best performing trials...")
        best_trials_data = get_sorted_trials(sort_by="mean_accuracy", ascending=False, n=3, thread_safe_state=thread_safe_state)
        if best_trials_data.get("status") == "completed":
            trials = best_trials_data.get("top_3_trials_by_mean_accuracy", [])
            best_trials_summary = json.dumps(trials, indent=2)
        else:
            best_trials_summary = "Not enough trial data to analyze."
    except Exception as e:
        best_trials_summary = f"Error analyzing trials: {e}"

    try:
        log("[Summary] Analyzing hyperparameter correlations...")
        correlation_data = get_correlation_matrix(thread_safe_state=thread_safe_state)
        if correlation_data.get("status") == "completed":
            correlations = correlation_data.get("correlation", {})
            correlation_summary = json.dumps(correlations.get("mean_accuracy", {}), indent=2)
        else:
            correlation_summary = "Not enough data to calculate correlations."
    except Exception as e:
        correlation_summary = f"Error analyzing correlations: {e}"

    final_summary = f"""
**Comprehensive Summary of Findings**

**1. Insights from Long-Term Memory:**
- {memory_summary}

**2. Top 3 Performing Configurations:**
```json
{best_trials_summary}
```

**3. Key Hyperparameter Correlations with Accuracy:**
```json
{correlation_summary}
```
"""
    return {"status": "completed", "summary": final_summary}


# ========================================================================


def propose_architectural_mutation() -> Dict[str, Any]:
    """Suggests a major architectural change if you are stuck."""
    log("Proposing an architectural mutation...")
    mutations = [
        "Replace standard Conv2D layers with DepthwiseSeparableConv2D layers to improve efficiency.",
        "Rebuild a convolutional block as a ResNet-style block by adding the input to the output of the block.",
        "Change the convolutional kernel size from (3, 3) to (5, 5) to increase the receptive field.",
        "Replace MaxPooling2D layers with a Conv2D layer using a stride of (2, 2) to learn the downsampling process.",
        "Add SpatialDropout2D after the convolutional blocks to prevent overfitting on feature maps.",
        "Ensure a BatchNormalization layer is placed after every Conv2D and Dense layer, before activation.",
        "Switch the activation function from ReLU to LeakyReLU to prevent the 'dying ReLU' problem.",
        "Switch the activation function from ReLU to GELU, as it is standard in modern high-performance models.",
    ]
    return {"status": "completed", "suggestion": random.choice(mutations)}


def _atomic_write(filepath: str, content: str, mode: str = "w") -> None:
    """
    Writes content to a temporary file and atomically replaces the target file.
    Prevents partial writes in case of process interruption.
    """
    dir_name = os.path.dirname(filepath)
    base_name = os.path.basename(filepath)
    temp_name = f".tmp_{base_name}_{int(time.time())}"
    temp_path = os.path.join(dir_name, temp_name)

    try:
        with open(temp_path, mode) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno()) # Ensure write to disk
        
        # Atomic replacement
        os.replace(temp_path, filepath) 
    
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass 
        raise e


def save_session(thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manually saves the session. The filename is generated automatically
    based on the current experiment name.
    """
    experiment_name = thread_safe_state.get("experiment_name", "autosave")
    filename = f"session_{experiment_name}.json"
    log(f"Saving session to {filename}...")
    try:
        data_to_save = {
            "results_log": thread_safe_state.get("results_log", []),
            "conversation_history": thread_safe_state.get("conversation_history", []),
        }
        content = json.dumps(data_to_save, indent=2)
        _atomic_write(filename, content)
        
        return {"status": "completed", "message": f"Session successfully saved to {filename}."}
    except Exception as e:
        return {"status": "error", "message": f"Save failed: {e}"}


def _detect_json_literals_in_python(code: str):
    """Return a list of (line, col, token) where bare JSON literals (null/true/false)
    appear as Python NAME tokens (i.e. NOT inside strings or comments).

    Uses tokenize so docstrings/comments containing the words don't false-positive.
    Returns empty list if the code is fine. Caller must ast.parse() the code FIRST
    so we don't have to handle tokenize-level failures here — by the time we run,
    the code is guaranteed syntactically valid Python.
    """
    import io as _io
    import tokenize as _tokenize
    bad = []
    forbidden = {"null", "true", "false"}
    tokens = _tokenize.tokenize(_io.BytesIO(code.encode("utf-8")).readline)
    for tok in tokens:
        if tok.type == _tokenize.NAME and tok.string in forbidden:
            bad.append((tok.start[0], tok.start[1], tok.string))
    return bad


def write_architecture_file(filename: str, code: str, thread_safe_state: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Writes the code of a Python model architecture to a specified file. The file must contain a function 'build_model(input_shape, num_classes, params)'. Use the same filename to overwrite and fix the code.
    IMPORTANT: Updates 'latest_created_architecture' in thread_safe_state to allow prompt injection.

    Validation (no silent auto-correction — the calling agent must see and fix the issue):
      1. Reject JSON literals (null/true/false) used as Python identifiers — common LLM mistake.
      2. Reject if the code does not parse as Python (ast.parse syntax error).
    """
    if not filename.endswith(".py"):
        return {"status": "error", "message": "Filename must end with .py"}

    if ".." in filename or "/" in filename:
        return {"status": "error", "message": "Invalid filename. Do not use relative paths or slashes."}

    # --- VALIDATION LAYER 1: ast.parse — must succeed before we look for bad names. ---
    # ast.parse confirms the code is syntactically valid Python; gives clean line:col on errors.
    try:
        import ast as _ast
        _ast.parse(code, filename=filename)
    except SyntaxError as _se:
        return {
            "status": "error",
            "message": (
                f"File '{filename}' NOT written: Python syntax error at line {_se.lineno} col {_se.offset}: "
                f"{_se.msg}. Fix the code and call write_architecture_file again."
            ),
        }

    # --- VALIDATION LAYER 2: bare JSON literals (null/true/false) ---
    # ast.parse accepts these because they ARE valid identifiers — they just refer to
    # nothing at runtime → NameError. Surface them to the agent with exact line:col so
    # it can fix and resubmit. Do NOT auto-correct: silent rewrites hide the bug from
    # the agent's learning loop (see memory: feedback_no_silent_fallbacks).
    bad_tokens = _detect_json_literals_in_python(code)
    if bad_tokens:
        details = ", ".join(f"line {ln} col {col}: '{tok}' (Python uses "
                            f"{'None' if tok == 'null' else 'True' if tok == 'true' else 'False'})"
                            for ln, col, tok in bad_tokens[:8])
        return {
            "status": "error",
            "message": (
                f"File '{filename}' NOT written: found {len(bad_tokens)} bare JSON literal(s) used as "
                f"Python identifiers (null/true/false). Python is case-sensitive and uses None/True/False. "
                f"Locations: {details}. Replace them and resubmit."
            ),
        }

    filepath = os.path.join(PERSISTENT_PATHS["custom_architectures_dir"], filename)
    try:
        _atomic_write(filepath, code)
        log(f"Architecture code successfully written to {filepath}")

        # --- NEW: Flag this as the latest created architecture ---
        if thread_safe_state is not None:
             thread_safe_state["latest_created_architecture"] = filename
             log(f"[State] Tagged '{filename}' as latest_created_architecture for Decider prompt.")

        return {"status": "completed", "message": f"File '{filename}' written/updated successfully."}
    except Exception as e:
        log(f"[TOOL ERROR] Writing file '{filename}' failed: {e}")
        return {"status": "error", "message": f"File write failed: {e}"}


# In tools.py


def validate_architecture_file(filename: str, expected_input_type: str, context_reason: str = "Strategic pivot to new architecture") -> Dict[str, Any]:
    """
    Lancia un processo worker separato per validare un file di architettura,
    specificando il tipo di input atteso ('1D', '2D', o '3D').
    
    Returns:
        Dict: Validation result. If successful, includes signals to switch architecture and trigger summary.
    """
    log(f"Delegating validation of '{filename}' to a worker, expecting '{expected_input_type}' input...")

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json", dir=PERSISTENT_PATHS["processed_data_dir"]
    ) as tmp:
        result_path = tmp.name
        worker_script_path = os.path.join(PERSISTENT_PATHS["scripts_dir"], "validation_worker.py")
    command = [
        "python",
        worker_script_path,
        "--filename",
        filename,
        "--output_path",
        result_path,
        "--expected_input_type",
        expected_input_type,
    ]

    log(f"Validation worker command: {' '.join(command)}")

    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
        with open(result_path, "r") as f:
            result = json.load(f)
            
        # --- SUCCESS HANDLING WITH PIVOT SIGNALS ---
        if result.get("status") == "completed":
             result["updated_selected_model"] = filename
             if context_reason:
                 result["trigger_summary_with_reason"] = context_reason
             log(f"[Validation Success] Attaching pivot signals for '{filename}': {context_reason}")
        # -------------------------------------------

    except subprocess.TimeoutExpired:
        result = {"status": "error", "message": "Validation timed out."}
    except subprocess.CalledProcessError as e:
        try:
            with open(result_path, "r") as f:
                result = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            result = {"status": "error", "message": f"Validation script failed: {e.stderr or e.stdout}"}
    except Exception as e:
        result = {"status": "error", "message": f"An unexpected error occurred: {e}"}
    finally:
        if os.path.exists(result_path):
            os.remove(result_path)
            
    return result

    log(f"Risultato ricevuto dal worker di validazione: {result.get('status')}")
    return result


def run_strategic_optuna_sweep(
    thread_safe_state: Dict[str, Any],
    n_trials: int = 20,
    hyperparameters: Optional[Dict] = None,
    experiment_name: str = "optuna_sweep",
    optimization_focus: str = "accuracy",
    manifest_name: str = "data_paper_mine_csv",
    trial_callback: Optional[Callable] = None,
    custom_architecture_file: Optional[str] = None,
    representation_type: Optional[str] = None,
    gpu_id: Optional[int] = None,
    save_model: bool = False,
) -> Dict[str, Any]:
    """
    Runs a full Optuna sweep. If using a custom architecture, 'representation_type' must be specified.
    """
    # Reduce Optuna verbosity to keep logs clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    log(f"Starting strategic Optuna sweep of {n_trials} trials for '{experiment_name}' on manifest '{manifest_name}' using GPU {gpu_id}.")

    if custom_architecture_file and not representation_type:
        log("[Tool Error] Missing 'representation_type' for custom architecture sweep.")
        return {
            "status": "error",
            "message": "Argument 'representation_type' is MANDATORY for sweeps with 'custom_architecture_file'. Specify the data format (e.g., '1D_CNN', '2D_GAF').",
        }

    effective_representation = representation_type if representation_type else experiment_name

    try:
        metadata = _load_dataset_metadata(thread_safe_state)
        if manifest_name not in metadata:
            log(f"[Auto-Recovery] Manifest '{manifest_name}' not found. Attempting to create it with default parameters...")
            try:
                create_dataset_manifest(manifest_name, {"representation_type": effective_representation}, thread_safe_state)
                metadata = _load_dataset_metadata(thread_safe_state)
                if manifest_name not in metadata:
                     return {"status": "error", "message": f"[SYSTEM_ERROR] Manifest '{manifest_name}' not found and auto-creation failed."}
                log(f"[Auto-Recovery] Manifest '{manifest_name}' successfully created.")
            except Exception as e:
                return {"status": "error", "message": f"[SYSTEM_ERROR] Manifest '{manifest_name}' not found and auto-creation crashed: {e}"}

        manifest_data = metadata[manifest_name]
        available_reps = list(manifest_data.get("generated_representations", {}).keys())

        if effective_representation not in manifest_data.get("generated_representations", {}):
            log(f"[Auto-Fix] Data representation '{effective_representation}' missing for '{manifest_name}'. Generating automatically...")
            
            # --- AUTO-GENERATION LOGIC (Ported from run_training_trial) ---
            gen_result = generate_representation(manifest_name, effective_representation, thread_safe_state)
            
            if gen_result.get("status") != "completed":
                return {
                    "status": "error", 
                    "message": f"Auto-generation failed for '{effective_representation}': {gen_result.get('message', 'Unknown error')}"
                }
            
            # Reload metadata to confirm registration
            metadata = _load_dataset_metadata(thread_safe_state)
            manifest_data = metadata[manifest_name]
            log(f"[Auto-Fix] Data successfully generated. Proceeding with Optuna Sweep.")
            
        data_path_root = manifest_data["path"]
    except Exception as e:
        log(f"[Tool Error] Failed during metadata/rep check: {e}")
        return {"status": "error", "message": f"Failed checking data representation: {traceback.format_exc()}"}

    if custom_architecture_file and not representation_type:
        return {
            "status": "error",
            "message": "Argument 'representation_type' is required when using a 'custom_architecture_file' in a sweep.",
        }

    if optimization_focus not in ["accuracy", "stability"]:
        return {"status": "error", "message": "Invalid 'optimization_focus'. Use 'accuracy' or 'stability'."}

    params_to_optimize = hyperparameters.copy() if hyperparameters else {}

    # --- Pre-Flight Guardrail Check ---
    architecture_name = custom_architecture_file if custom_architecture_file else thread_safe_state.get("selected_model", "Unknown")
    past_trials = _load_optuna_log(architecture_name, manifest_name)
    blocked, warning_msg = _check_search_space_overlap(params_to_optimize, past_trials)
    if blocked:
        log(f"[Optuna Guardrail] Sweep blocked: {warning_msg}")
        return {
            "status": "guardrail_blocked",
            "message": warning_msg,
            "past_trials_summary": {
                "total_past_trials": len(past_trials),
                "architecture": architecture_name,
                "manifest": manifest_name
            }
        }
    log(f"[Optuna Guardrail] Sweep allowed ({len(past_trials)} past trials for {architecture_name}).")
    # ----------------------------------
    
    def objective(trial):
        params = {}
        for name, definition in params_to_optimize.items():
            if not isinstance(definition, dict):
                continue
            param_type = definition.get("type")
            if param_type == "int":
                params[name] = trial.suggest_int(name, definition["min"], definition["max"])
            elif param_type == "float":
                params[name] = trial.suggest_float(name, definition["min"], definition["max"])
            elif param_type == "loguniform":
                params[name] = trial.suggest_float(name, definition["min"], definition["max"], log=True)
            elif param_type == "choice":
                params[name] = trial.suggest_categorical(name, definition["choices"])

        trial_hyperparameters = params.copy()

        # Apply sensible fallback defaults ONLY for params not in the search space.
        # Priority: (1) Optuna-suggested value, (2) explicit fallback below, (3) worker silent default.
        # epochs and patience must NOT be hardcoded — if they are not in the search space,
        # use a conservative fallback that does not truncate training prematurely.
        _optuna_fallbacks = {
            "epochs":       100,   # longer than old 50 — early stopping controls actual duration
            "patience":     20,    # wider window than old 15 — avoids premature stop
            "batch_norm":   1,     # always enable — worker default=0 causes silent underfitting
            "weight_decay": 0.0,   # explicit zero is logged; omitting it is invisible
        }
        for _k, _v in _optuna_fallbacks.items():
            if _k not in trial_hyperparameters:
                trial_hyperparameters[_k] = _v

        # For Optuna Sweeps we MUST use sweep_gpu_id (which is None) 
        # to allow Celery tasks to acquire free GPUs dynamically
        result = run_training_trial(
            experiment_name=experiment_name,
            manifest_name=manifest_name,
            custom_architecture_file=custom_architecture_file,
            representation_type=representation_type,
            thread_safe_state=thread_safe_state, # Pass thread_safe_state
            gpu_id=None, # Pass None so Celery workers dynamically acquire locks
            save_model=save_model, # Disable model saving for optimization sweep
            **trial_hyperparameters,
        )

        if result.get("status") != "completed":
            err_msg = result.get("message", "unknown error")
            log(f"[Optuna] Trial failed. Returning penalty score. Error: {err_msg}")
            # Save error in user_attrs so the post-sweep summary can surface it
            trial.set_user_attr("error", str(err_msg)[:500])
            trial.set_user_attr("failed", True)
            return 0.0

        mean_acc = result.get("mean_accuracy", 0.0)
        inference_time = result.get("inference_time_ms", 9999.0) # Default high if missing

        # --- Extract Held-Out Test Accuracy ---
        held_out_acc = result.get("held_out_test_accuracy")

        # NO fallback by design: if the held-out evaluation failed, this trial scores 0.0
        # and Optuna ranks it accordingly. The agent is self-healing — it must SEE the
        # failure (with the underlying worker error) to fix the root cause. A charity
        # fallback to CV accuracy would let buggy trials look healthy and the agent would
        # never learn to repair the broken evaluation path.
        if held_out_acc is None or held_out_acc == 0.0:
            eval_err = result.get("evaluation_error") or "unknown (file missing or worker did not report an error string)"
            log(f"[Optuna FAILURE] Trial {trial.number}: held_out_test_accuracy={held_out_acc}. "
                f"Scoring 0.0. Root cause from worker: {eval_err}")
            held_out_acc = 0.0
        # ----------------------------------------

        trial.set_user_attr("mean_accuracy", mean_acc)
        trial.set_user_attr("inference_time_ms", inference_time)
        trial.set_user_attr("held_out_test_accuracy", held_out_acc)

        # Optimize held_out_test_accuracy (true generalization) not val CV accuracy
        return held_out_acc

    # Compute coverage warning once in outer scope (accessible to report builder below)
    _RECOMMENDED_SWEEP_PARAMS = {
        "learning_rate", "dropout_rate", "filters", "kernel_size",
        "batch_size", "epochs", "patience", "weight_decay", "batch_norm"
    }
    _missing_from_sweep = _RECOMMENDED_SWEEP_PARAMS - set(params_to_optimize.keys())
    if _missing_from_sweep:
        log(f"[Optuna Coverage Warning] These params are NOT in the search space and will use fallback defaults: {sorted(_missing_from_sweep)}. Consider adding them to the search space for better exploration.")

    # Single-objective study: Maximize held-out test accuracy (true generalization metric)
    study = optuna.create_study(direction="maximize")

    # --- CENTRALIZED GPU ALLOCATION ---
    # For Optuna Sweeps on the cluster, we MUST distribute trials across all available GPUs.
    # If the orchestrator passed a specific gpu_id, we ignore it here so Celery can auto-acquire free GPUs for each trial.
    sweep_gpu_id = None

    if sweep_gpu_id is not None:
        n_jobs = 1
        log(f"[Optuna Strategy] Running sequentially on orchestrator-assigned GPU {sweep_gpu_id}.")
    else:
        # Cluster mode: Let Celery and tasks.py manage available GPUs via locking.
        n_jobs = 6  # fallback
        cfg = thread_safe_state.get("cfg")
        if cfg:
            exec_cfg = cfg.get("execution") if isinstance(cfg, dict) else getattr(cfg, "execution", None)
            if exec_cfg:
                n_jobs = exec_cfg.get("max_concurrent_gpu", 6) if isinstance(exec_cfg, dict) else getattr(exec_cfg, "max_concurrent_gpu", 6)

        # Override with env var if explicitly set
        n_jobs = int(os.environ.get("MAX_CONCURRENT_TRIALS", n_jobs))
        log(f"[Optuna Strategy] Running concurrently with {n_jobs} parallel tasks across cluster GPUs (Agent requested GPU {gpu_id} bypassed).")

    try:
        # n_jobs ensures we utilize all allocated resources in parallel.
        # run_training_trial dispatches to Celery, so this loop mainly manages the futures.
        # We limit n_jobs to MAX_CONCURRENT_TRIALS to match Celery worker count.
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        
    except Exception as e:
        log(f"[Tool Error] Optuna sweep failed: {e}")
        return {"status": "error", "message": f"Optuna sweep failed: {traceback.format_exc()}"}

    log("Sweep complete. Building strategic report...")
    
    log("Sweep complete. Building strategic report...")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        # Collect failure reasons from all failed trials so the Decider can diagnose the root cause
        failed_trials = [t for t in study.trials if t.state != optuna.trial.TrialState.COMPLETE]
        failure_details = []
        for ft in failed_trials[:5]:  # Cap at 5 to avoid token bloat
            attrs = ft.user_attrs or {}
            err = attrs.get("error", "") or attrs.get("message", "")
            if not err and hasattr(ft, "exception") and ft.exception:
                err = str(ft.exception)
            if err:
                failure_details.append(f"  Trial {ft.number}: {err}")
        failure_summary = "\n".join(failure_details) if failure_details else "  (no error details captured in trial user_attrs)"
        return {
            "status": "error",
            "message": (
                f"No trials were completed successfully ({len(failed_trials)} failed). "
                f"Root cause — check the FIRST trial error below and fix the call before retrying:\n"
                f"{failure_summary}\n"
                f"HINT: If error mentions 'expects 4D input' or 'representation_type', "
                f"the representation is incompatible with the custom architecture. "
                f"Use '3D_DYNAMIC_GAF', '3D_DYNAMIC_CWT', or '3D_WAVELET_CWT' with hybrid_resnet_gaf_ltc.py."
            ),
        }

    # Single-objective study: best_trial is the trial with highest held_out_test_accuracy
    try:
        best_trial_in_study = study.best_trial
        best_value = best_trial_in_study.value
    except Exception as e:
        log(f"[Optuna Report Warning] Failed to determine best trial: {e}")
        # Fallback: manually sort by held_out_test_accuracy user attr
        completed_trials.sort(key=lambda t: t.user_attrs.get("held_out_test_accuracy", 0.0), reverse=True)
        best_trial_in_study = completed_trials[0]
        best_value = best_trial_in_study.user_attrs.get("held_out_test_accuracy", 0.0)

    all_values = {t.value if t.value is not None else 0.0 for t in completed_trials}
    
    if len(all_values) <= 1:
        log("[Optuna WARNING] All completed trials have the same value. Skipping parameter importance analysis.")
        
        report = {
            "sweep_summary": {
                "total_trials_requested": n_trials,
                "completed_trials": len(completed_trials),
                "focus": optimization_focus,
                "best_value_achieved": best_value,
                "note": "All trials resulted in the same value; parameter importance could not be calculated.",
            },
            "best_trial_by_accuracy": {
                "params": best_trial_in_study.params,
                "mean_accuracy": best_trial_in_study.user_attrs.get("mean_accuracy", best_value),
            },
        }
        
        # --- SAVE REPORT TO DISK (Early Exit) ---
        try:
            reports_dir = os.path.join(PERSISTENT_PATHS.get("processed_data_dir", "processed_data"), "research_reports")
            os.makedirs(reports_dir, exist_ok=True)
            report_filename = f"optuna_report_{experiment_name}_{int(time.time())}.json"
            report_path = os.path.join(reports_dir, report_filename)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
            log(f"[Optuna] Strategic Report saved to: {report_path}")
            report["report_path"] = report_path
        except Exception as e:
            log(f"[Optuna Warning] Failed to save report to disk: {e}")
        # ----------------------------------------
        
        return {"status": "completed", "strategic_report": report}

    best_params = best_trial_in_study.params

    full_best_trial_data = {}
    # We retrieve the results_log from the thread_safe_state as it would have been updated by run_training_trial
    current_results_log = thread_safe_state.get("results_log", [])
    for entry in reversed(current_results_log):
        # Simple heuristic matching to find the full record
        if entry.get("source") == "agent_trial" and entry.get("params") == best_params:
            full_best_trial_data = entry
            break

    full_best_trial_data.pop("architecture", None)
    full_best_trial_data.pop("manifest_name", None)
    full_best_trial_data.pop("source", None)

    # Safe Parameter Importance Calculation
    param_importances = {}
    try:
        param_importances = get_param_importances(study)
    except Exception as e:
        log(f"[Optuna Warning] Parameter importance failed: {e}")
        param_importances = {"error": str(e)}

    # ─── OPTUNA GUARDRAIL: Append completed trials ────
    records_to_log = []
    for t in completed_trials:
        records_to_log.append({
            "timestamp": int(time.time()),
            "architecture": architecture_name,
            "manifest_name": manifest_name,
            "experiment_name": experiment_name,
            "params": t.params,
            "mean_accuracy": t.user_attrs.get("mean_accuracy", t.values[0] if t.values else 0.0),
            "inference_time_ms": t.user_attrs.get("inference_time_ms", t.values[1] if (t.values and len(t.values) > 1) else 9999.0),
            "held_out_test_accuracy": t.user_attrs.get("held_out_test_accuracy", 0.0)
        })
    _append_to_optuna_log(records_to_log)
    # ──────────────────────────────────────────────────

    # --- Enhanced Strategic Report with Held-Out Accuracy Data ---
    completed_trials_summary = []
    best_held_out_trial = None
    max_held_out_acc = -1.0

    for t in completed_trials:
        t_held_out = t.user_attrs.get("held_out_test_accuracy", 0.0)
        t_mean_acc = t.user_attrs.get("mean_accuracy", t.values[0] if t.values else 0.0)
        
        trial_record = {
            "trial_number": t.number,
            "params": t.params,
            "mean_accuracy": t_mean_acc,
            "held_out_test_accuracy": t_held_out,
            "inference_time_ms": t.user_attrs.get("inference_time_ms", t.values[1] if (t.values and len(t.values) > 1) else 9999.0)
        }
        completed_trials_summary.append(trial_record)

        if t_held_out > max_held_out_acc:
            max_held_out_acc = t_held_out
            best_held_out_trial = trial_record

    report = {
        "sweep_summary": {
            "total_trials_requested": n_trials,
            "completed_trials": len(completed_trials),
            "focus": optimization_focus,
            "best_value_achieved": best_value,
            "params_not_in_search_space": sorted(_missing_from_sweep) if _missing_from_sweep else [],
            "coverage_warning": (
                f"⚠️ These params used fallback defaults (not swept): {sorted(_missing_from_sweep)}. "
                f"Add them to the search space in the next sweep for better exploration."
            ) if _missing_from_sweep else None,
        },
        "best_trial_metrics": full_best_trial_data,
        "best_trial_by_held_out_accuracy": best_held_out_trial,
        "parameter_importances": param_importances,
        "all_trials_summary": completed_trials_summary
    }

    # --- SAVE REPORT TO DISK ---
    try:
        reports_dir = os.path.join(PERSISTENT_PATHS.get("processed_data_dir", "processed_data"), "research_reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_filename = f"optuna_report_{experiment_name}_{int(time.time())}.json"
        report_path = os.path.join(reports_dir, report_filename)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        log(f"[Optuna] Strategic Report saved to: {report_path}")
        report["report_path"] = report_path # Return path to agent
    except Exception as e:
        log(f"[Optuna Warning] Failed to save report to disk: {e}")
    # ---------------------------

    return {"status": "completed", "strategic_report": report}
def flag_architecture(architecture_name: str, flag: str, thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Manually flags an architecture to stop optimization on it.
    This is the agent's most important tool for preventing loops.
    Valid flags are:
    - 'completed': The model has met the optimization target.
    - 'abandoned_underperforming': The model is stuck in a plateau and should be ignored.
    """
    log(f"[STRATEGY] Setting manual flag for '{architecture_name}' to '{flag}'")
    valid_flags = ["completed", "abandoned_underperforming"]
    if flag not in valid_flags:
        return {"status": "error", "message": f"Invalid flag. Must be one of: {valid_flags}"}

    # Ensure the dictionary exists in the state
    if "architecture_flags" not in thread_safe_state:
        thread_safe_state["architecture_flags"] = {}
    
    # Update the GLOBAL state in place (not a copy)
    thread_safe_state["architecture_flags"][architecture_name] = flag
    
    # Force immediate persistence to disk
    try:
        save_state(thread_safe_state)
        log(f"[STRATEGY] Flag saved to disk: {architecture_name} -> {flag}")
    except Exception as e:
        log(f"[STRATEGY ERROR] Failed to save state after flagging: {e}")
    
    # Return the updated flags to be synced back to st.session_state
    return {
        "status": "completed",
        "message": f"Flag for '{architecture_name}' has been set to '{flag}' and saved to disk. Call 'get_optimization_status' to see the updated list and 'set_optimization_model' to switch to a new 'pending' model.",
        "updated_architecture_flags": thread_safe_state["architecture_flags"]
    }


# --- FUNZIONE SOSTITUITA: get_optimization_status ---
def get_optimization_status(thread_safe_state: Dict[str, Any], optimization_threshold: float = 0.95, analysis_window: int = 10) -> Dict[str, Any]:
    """
    Analyzes and reports the optimization 'health' of all architectures.
    """
    # Use optimization_threshold from thread_safe_state if available, otherwise use default
    optimization_threshold = thread_safe_state.get("optimization_target", optimization_threshold)

    log(f"Analyzing optimization status. Threshold: {optimization_threshold}, Window: {analysis_window}")

    all_known_architectures = set(MODEL_BUILDERS.keys())
    results_log = thread_safe_state.get("results_log", [])
    df = pd.DataFrame(results_log) if results_log else pd.DataFrame()
    if not df.empty and "architecture" in df.columns:
        all_known_architectures.update(df["architecture"].unique())

    manual_flags = thread_safe_state.get("architecture_flags", {})

    final_report = {}
    pending_architectures_list = []

    for arch_name in all_known_architectures:
        manual_flag = manual_flags.get(arch_name, "pending")

        health_report = {
            "best_accuracy": 0.0,
            "steps_since_new_best": 0,
            "recent_trend_slope": 0.0,
            "recent_std_dev": 0.0,
            "total_trials_for_model": 0,
            "auto_trend_status": "N/A",
        }

        arch_df = pd.DataFrame()
        if not df.empty and "architecture" in df.columns and "held_out_test_accuracy" in df.columns:
            arch_df = df[df["architecture"] == arch_name].copy()
        elif not df.empty and "architecture" in df.columns and "mean_accuracy" in df.columns:
            arch_df = df[df["architecture"] == arch_name].copy()

        if not arch_df.empty:
            # Use held_out_test_accuracy as the primary metric if available
            if "held_out_test_accuracy" in arch_df.columns and arch_df["held_out_test_accuracy"].max() > 0:
                all_accuracies = arch_df["held_out_test_accuracy"].tolist()
            else:
                all_accuracies = arch_df["mean_accuracy"].tolist()
            if not all_accuracies:
                continue
            best_acc_ever = float(np.max(all_accuracies))
            best_acc_position = int(np.argmax(np.array(all_accuracies)))
            total_trials = int(len(all_accuracies))

            health_report["best_accuracy"] = best_acc_ever
            health_report["total_trials_for_model"] = total_trials
            health_report["steps_since_new_best"] = (total_trials - 1) - best_acc_position

            if manual_flag == "pending":
                # UPDATED: More conservative thresholds (3x increase) to prevent premature abandonment
                if total_trials < 15:  # Was 5, now 15 - more data before plateau detection
                    health_report["auto_trend_status"] = "initializing"
                else:
                    recent_accuracies = np.array(all_accuracies[-analysis_window:])
                    if len(recent_accuracies) >= 2:
                        X_recent = np.arange(len(recent_accuracies)).reshape(-1, 1)
                        try:
                            model = LinearRegression()
                            model.fit(X_recent, recent_accuracies)
                            slope = model.coef_[0]
                            std_dev = np.std(recent_accuracies)

                            health_report["recent_trend_slope"] = slope
                            health_report["recent_std_dev"] = std_dev
                            steps_stuck = health_report["steps_since_new_best"]

                            if best_acc_ever >= optimization_threshold:
                                health_report["auto_trend_status"] = "optimized"
                            elif slope > 0.005:
                                health_report["auto_trend_status"] = "strong_growth"
                            elif slope > 0.001:
                                health_report["auto_trend_status"] = "slow_growth"
                            # UPDATED: 15-25 steps (was 5-10) before suggesting advanced strategies
                            elif steps_stuck >= 15 and steps_stuck <= 25 and slope > -0.001:
                                health_report["auto_trend_status"] = "plateauing_try_harder"
                            # UPDATED: 25+ steps (was 10+) before suggesting innovation/Supernet
                            elif (
                                steps_stuck > 25 and slope <= 0.001 and (len(recent_accuracies) < 3 or std_dev < 0.01)
                            ):
                                health_report["auto_trend_status"] = "plateauing_innovate"
                            elif slope <= -0.001:
                                health_report["auto_trend_status"] = "in_decline"
                            else:
                                health_report["auto_trend_status"] = "uncertain"

                        except Exception as e:
                            log(f"[Tool Error] Trend analysis failed for {arch_name}: {e}")
                            health_report["auto_trend_status"] = "analysis_failed"
                    else:
                        health_report["auto_trend_status"] = "initializing"
            else:
                health_report["auto_trend_status"] = "N/A"

        final_report[arch_name] = {
            "manual_flag": manual_flag,
            "target_threshold": optimization_threshold,
            "performance_health_report": health_report,
        }

        if manual_flag == "pending":
            pending_architectures_list.append(arch_name)

    return {
        "status": "completed",
        "optimization_summary": final_report,
        "pending_architectures": pending_architectures_list,
    }


# In tools.py


def set_optimization_model(model_name: str, thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sets the model architecture to be optimized in the next steps.
    This tool MUST be called after using 'get_optimization_status' to switch to a pending architecture.
    """
    log(f"[STRATEGY] Agent requested to switch optimization focus to: {model_name}")

    from tools import MODEL_BUILDERS as local_model_builders

    base_models = list(local_model_builders.keys())

    custom_models = []
    arch_dir = PERSISTENT_PATHS.get("custom_architectures_dir")

    if arch_dir and os.path.exists(arch_dir):
        try:
            for f in os.listdir(arch_dir):
                if f.endswith(".py"):
                    custom_models.append(f)
        except Exception as e:
            log(f"[Tool set_optimization_model] WARNING: Could not scan architectures directory: {e}")

    all_models = list(dict.fromkeys(base_models + custom_models))

    if model_name not in all_models:
        log(f"[TOOL ERROR] Invalid model name '{model_name}'. Full list: {all_models}")
        return {"status": "error", "message": f"Invalid model name '{model_name}'. Available models are: {all_models}"}
    
    log(f"[STRATEGY] Archiving conversation for '{thread_safe_state.get('selected_model')}' before switching.")
    archive_conversation_log(thread_safe_state, thread_safe_state.get("conversation_history", []))

    updated_conversation_history = [] # Clear conversation history
    updated_selected_model = model_name

    log("[STRATEGY] Focus switched successfully. Resetting conversation for new target.")

    return {
        "status": "completed",
        "message": f"Optimization model switched to '{model_name}'. The conversation history has been cleared to focus on the new task.",
        "updated_selected_model": updated_selected_model,
        "updated_conversation_history": updated_conversation_history,
    }



def list_available_tools(thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:
    """Lists all available tools with their names and a short description."""
    log("Listing all available tools for the agent...")
    tools_info = {}
    for tool_name, tool_func in AVAILABLE_TOOLS.items():
        docstring = None
        
        # 1. Try to find the function in globals() by name (since keys match function names)
        # This bypasses the lambda wrapper which has no docstring
        real_func = globals().get(tool_name)
        if real_func and hasattr(real_func, "__doc__") and real_func.__doc__:
            docstring = real_func.__doc__
            
        # 2. Fallback: Check the tool_func itself (in case it's not a lambda)
        if not docstring and hasattr(tool_func, "__doc__") and tool_func.__doc__:
            docstring = tool_func.__doc__
            
        if docstring:
            first_line = docstring.strip().split("\n")[0]
            tools_info[tool_name] = first_line
        else:
            tools_info[tool_name] = "No description available."

    return {"status": "completed", "available_tools": tools_info}


def get_architecture_summary(
    experiment_name: Optional[str] = None, custom_architecture_file: Optional[str] = None, thread_safe_state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generates and returns a text summary of a model's architecture.
    Use this to "see" the model's layers. Specify either a standard experiment_name or a custom_architecture_file.
    """
    log("Generating architecture summary...")
    # --- RELAXED VALIDATION ---
    if not experiment_name and not custom_architecture_file:
         return {
            "status": "error",
            "message": "Specify at least one of 'experiment_name' or 'custom_architecture_file'.",
        }
    
    # If both are present, prioritize custom_architecture_file as it is more specific
    if custom_architecture_file and experiment_name:
        log(f"[Tool] both provided. Prioritizing custom file: {custom_architecture_file}")
    # --------------------------

    model = None
    try:
        model_builder = None

        context_key = thread_safe_state.get("selected_model") or experiment_name or custom_architecture_file

        if "1D" in context_key:
            dummy_shape = (128, 1)
        elif "3D" in context_key:
            dummy_shape = (10, 64, 64, 1)
        else:  # Default to 2D
            dummy_shape = (64, 64, 1)

        log(f"Inferred a {len(dummy_shape)}D dummy shape based on context: '{context_key}' -> {dummy_shape}")

        if custom_architecture_file:
            filepath = os.path.join(PERSISTENT_PATHS["custom_architectures_dir"], custom_architecture_file)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Custom architecture file '{custom_architecture_file}' not found.")

            spec = importlib.util.spec_from_file_location(custom_architecture_file.replace(".py", ""), filepath)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
            model_builder = custom_module.build_model
        else:
            if experiment_name not in MODEL_BUILDERS:
                raise ValueError(f"Model builder for '{experiment_name}' not found.")
            model_builder = MODEL_BUILDERS[experiment_name]

        dummy_params = {
            "learning_rate": 1e-3,
            "dropout_rate": 0.5,
            "filters": 32,
            "kernel_size": 3,
            "dense_units": 64,
            "num_conv_layers": 2,
        }
        model = model_builder(input_shape=dummy_shape, num_classes=10, params=dummy_params)

        string_buffer = io.StringIO()
        with redirect_stdout(string_buffer):
            model.summary(expand_nested=True)
        model_summary_str = string_buffer.getvalue()

        log("Summary generated successfully.")
        return {"status": "completed", "summary": model_summary_str}

    except Exception:
        error_trace = traceback.format_exc()
        log(f"[TOOL ERROR] Summary creation failed:\n{error_trace}")
        return {"status": "error", "message": f"Summary creation failed: {error_trace}"}
    finally:
        if model is not None:
            del model
        K.clear_session()
        gc.collect()


# tools.py


# def analyze_successful_strategy(action_name: Optional[str] = None) -> Dict[str, Any]:
#     """
#     Analizza il grafo strategico per trovare le transizioni stato-azione di maggior successo
#     e recupera le scoperte associate dalla memoria semantica. È uno strumento metacognitivo.
#     Se viene fornito 'action_name', analizza quello strumento specifico. Altrimenti, trova
#     la migliore strategia complessiva.
#     """
#     log(f"[METACOGNITION] Analyzing successful strategies for: {action_name or 'Best Overall'}")
#     graph = st.session_state.strategic_graph_memory.graph

#     if not graph or graph.number_of_edges() == 0:
#         return {"status": "completed", "report": "The strategic map is empty. No strategies to analyze."}

#     # Ottiene tutti gli archi con i loro dati completi
#     edges_to_analyze = list(graph.edges(data=True, keys=True))

#     # Se l'utente vuole analizzare un'azione specifica, filtriamo gli archi
#     if action_name:
#         edges_to_analyze = [e for e in edges_to_analyze if e[3].get('action_name') == action_name]
#         if not edges_to_analyze:
#             return {"status": "completed", "report": f"No experiences found for the action '{action_name}'."}

#     try:
#         # Trova l'arco con la ricompensa media più alta (ora il dato esiste!)
#         # e[0] = nodo_partenza, e[1] = nodo_arrivo, e[3] = dati_arco
#         best_edge = max(edges_to_analyze, key=lambda edge: edge[3].get('avg_reward', -float('inf')))
#     except ValueError:
#         return {"status": "completed", "report": "No valid strategies found to analyze."}

#     start_node, end_node, _, edge_data = best_edge
#     action = edge_data.get('action_name')
#     reward = edge_data.get('avg_reward')
#     finding_ids = edge_data.get('finding_ids', [])

#     if not finding_ids:
#         report = (f"The most successful strategy for '{action}' (from '{start_node}' to '{end_node}') has an average reward of {reward:.2f}. "
#                   "However, no specific semantic memories are linked to its success yet.")
#         return {"status": "completed", "report": report}

#     # Recupera le scoperte testuali dalla memoria usando gli ID
#     linked_findings = []
#     all_memories = st.session_state.memory.memory
#     for mem in all_memories:
#         if mem.get('finding_id') in finding_ids:
#             linked_findings.append(mem.get('finding'))

#     findings_text = "\n- ".join(linked_findings) if linked_findings else "N/A"

#     report = (f"**Metacognitive Analysis Report for '{action}'**\n\n"
#               f"**Most Successful Application:**\n"
#               f"- **Transition**: From `{start_node}` to `{end_node}`\n"
#               f"- **Average Reward**: `{reward:.2f}`\n\n"
#               f"**Associated Discoveries (The 'Why'):**\n"
#               f"This success is linked to these findings:\n"
#               f"- {findings_text}")


#     return {"status": "completed", "report": report}
def propose_architecture_upgrade(
    agent: Any, thread_safe_state: Dict[str, Any], base_file: str, directive: str, **kwargs
) -> Dict[str, Any]:
    """
    MUTATION entry point: upgrade an EXISTING custom architecture file in place of
    generating one from scratch. Reads `base_file`, asks the Innovation Team to apply
    `directive` as a targeted modification, then runs the standard validation +
    smoke-test + registration pipeline (reuses propose_intelligent_architecture).

    MVP (Path A): no fitness gate — produces a validated candidate; the Theorist tests
    it with run_training_trial. The parent file is never modified (a NEW file is written).
    """
    if not base_file or not isinstance(base_file, str) or not base_file.endswith(".py"):
        return {"status": "error",
                "message": f"base_file must be an existing custom architecture .py filename (got: {base_file!r})."}
    arch_dir = PERSISTENT_PATHS.get("custom_architectures_dir", "")
    if arch_dir and not os.path.exists(os.path.join(arch_dir, base_file)):
        return {"status": "error",
                "message": f"base_file '{base_file}' not found in custom architectures dir ({arch_dir})."}
    if not directive or not isinstance(directive, str) or not directive.strip():
        return {"status": "error", "message": "directive (the improvement instruction) is required and must be non-empty."}

    log(f"[TOOL] Architecture UPGRADE requested: base='{base_file}' directive='{directive[:120]}'")

    # Delegate to the generation pipeline, pinning the base file and using the directive
    # as the hypothesis. base_file flows through to InnovationCollaborator.run_innovation_cycle.
    kwargs.pop("hypothesis", None)
    kwargs.pop("architecture_context", None)
    kwargs.pop("base_file", None)
    return propose_intelligent_architecture(
        agent=agent,
        thread_safe_state=thread_safe_state,
        hypothesis=directive.strip(),
        architecture_context=base_file,
        base_file=base_file,
        **kwargs,
    )


def propose_intelligent_architecture(
    agent: Any, thread_safe_state: Dict[str, Any], architecture_filter: Optional[str] = None, **kwargs
) -> Dict[str, Any]:
    """
    Analyzes the entire experiment history to intelligently propose a novel model architecture.
    Uses the InnovationCollaborator team to handle the full lifecycle (Scout -> Code -> Validate -> Train).
    """
    # Import locale per evitare dipendenze circolari se necessario, o assicurati che sia importato in alto
    try:
        from agent import InnovationCollaborator
    except ImportError:
        # Fallback o gestione errore se agent non è accessibile
        return {"status": "error", "message": "Could not import InnovationCollaborator from agent."}
    
    log("[TOOL] Starting intelligent architecture proposal via InnovationCollaborator...")
    
    if not agent:
        return {"status": "error", "message": "Agent instance not provided."}

    # Prepare context
    results_log = thread_safe_state.get("results_log", [])
    selected_model = thread_safe_state.get("selected_model", "Unknown")
    
    # Allow override for specific file context (e.g. from Supernet)
    architecture_context = kwargs.get("architecture_context", selected_model)
    
    # Simple hypothesis generation (or use Strategist if available)
    initial_hypothesis = kwargs.get("hypothesis", f"Improve {selected_model} using advanced techniques.")
    
    # Retrieve the current recursion depth (injected by main.py)
    initial_depth = kwargs.get("depth", 0)

    # Define execute_tool_func wrapper
    def recursive_execute_tool(name, args, depth_offset=0):
        current_depth = initial_depth + depth_offset
        MAX_DEPTH = 5
        
        if current_depth > MAX_DEPTH:
            return {"status": "error", "message": f"Recursion limit ({MAX_DEPTH}) exceeded in intelligent architecture proposal."}

        if name in AVAILABLE_TOOLS:
            # Inject state if missing
            if "thread_safe_state" not in args:
                args["thread_safe_state"] = thread_safe_state
            
            # Prevent infinite recursion or specialized handling
            if name == "propose_intelligent_architecture":
                args["agent"] = agent
            
            TOOLS_REQUIRING_EXECUTE_FUNC = [
                "run_architecture_comparison", 
                "run_strategic_graph_evaluation", 
                "generate_scientific_report", 
                "generate_finetuning_dataset"
            ]
            if name in TOOLS_REQUIRING_EXECUTE_FUNC:
                 # Create a closure that increments the depth offset
                 def depth_aware_inner(n, a):
                     return recursive_execute_tool(n, a, depth_offset=depth_offset + 1)
                 args["execute_tool_func"] = depth_aware_inner

            try:
                return AVAILABLE_TOOLS[name](**args)
            except Exception as e:
                return {"status": "error", "message": f"Tool execution failed: {e}"}
        else:
            return {"status": "error", "message": f"Tool {name} not found."}

    try:
        coding_model_name = thread_safe_state.get("coding_model_name")
        coding_agent = thread_safe_state.get("coding_agent")
        
        if coding_agent:
             log(f"[TOOL] Innovation Team using specialized Coding Agent (Model: {coding_agent.model_name})")
        elif coding_model_name:
            log(f"[TOOL] Innovation Team using specialized model override: {coding_model_name}")

        collaborator = InnovationCollaborator(agent, coding_model_name=coding_model_name, coding_agent=coding_agent)
        final_thoughts, result = collaborator.run_innovation_cycle(
            system_prompt="You are an expert AI Architect.",
            conversation_history=[],
            initial_hypothesis=initial_hypothesis,
            architecture_context=architecture_context,
            execute_tool_func=recursive_execute_tool,
            model_type=selected_model,
            metadata_info=f"Targeting {selected_model}. History length: {len(results_log)}",
            thread_safe_state=thread_safe_state,
            base_file=kwargs.get("base_file"),
        )
        
        response = {
            "status": "completed",
            "message": "Innovation cycle completed.",
            "final_thoughts": final_thoughts,
            "result": result,
        }
        # Promote architecture metadata to top level for easy capture by main.py.
        if isinstance(result, dict):
            if result.get("architecture_file"):
                response["architecture_file"] = result["architecture_file"]
            if result.get("representation_type"):
                response["representation_type"] = result["representation_type"]
            if result.get("manifest_name"):
                response["manifest_name"] = result["manifest_name"]

        # B1: Auto-switch to the new architecture so the next cycle uses it immediately.
        # This prevents small LLMs from ignoring the innovation result.
        if isinstance(result, dict) and result.get("architecture_file"):
            arch_file = result["architecture_file"]
            rep_type = result.get("representation_type", "")
            try:
                switch_result = recursive_execute_tool("set_optimization_model", {"model_name": arch_file})
                log(f"[TOOL] Auto-switched to new architecture '{arch_file}': {switch_result.get('message', '')}")
                response["auto_switch"] = switch_result.get("status", "unknown")
                # FIX B1a: apply the switch directly to thread_safe_state — recursive_execute_tool
                # bypasses main.py's state-interception layer so updated_selected_model is never applied.
                if switch_result.get("updated_selected_model"):
                    thread_safe_state["selected_model"] = switch_result["updated_selected_model"]
                    log(f"[TOOL] thread_safe_state['selected_model'] updated to '{switch_result['updated_selected_model']}'")
                # FIX B1b: propagate updated_selected_model to the top-level response so main.py's
                # execute_tool can also apply it (handles the case where propose_intelligent_architecture
                # is called directly by the Decider via main.py, not recursively).
                if switch_result.get("updated_selected_model"):
                    response["updated_selected_model"] = switch_result["updated_selected_model"]
            except Exception as se:
                log(f"[TOOL] WARNING: Auto-switch failed for '{arch_file}': {se}")
            try:
                recursive_execute_tool("memorize_finding", {
                    "finding": (
                        f"Innovation Team built '{arch_file}' (rep: {rep_type}). "
                        # default=str so pd.Interval (from pd.qcut in analyze_hyperparameter_tradeoffs)
                        # and any other non-JSON-native type in `result` serialize via __str__ rather
                        # than raising TypeError and breaking the entire smoke-test report.
                        f"Smoke test result: {json.dumps(result, default=str)}"
                    ),
                    "experiment_name": arch_file,
                    "representation_type": rep_type,
                    "category": "innovation_result",
                })
            except Exception as me:
                log(f"[TOOL] WARNING: memorize_finding after innovation failed: {me}")

        return response
    except Exception as e:
        return {"status": "error", "message": f"Innovation cycle failed: {traceback.format_exc()}"}


def query_research_archive(query: str, top_k: int = 3, thread_safe_state: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Searches the RAG database of past scientific reports and web research.
    Returns the most relevant chunks of text to act as literature review context.
    """
    from cluster.rag_system import TfIdfRetriever
    
    log(f"[Tool Trigger] Initiating 'query_research_archive' for query: {query}")
    
    reports_dir = os.path.join(PERSISTENT_PATHS.get("processed_data_dir", "processed_data"), "research_reports")
    
    if not os.path.exists(reports_dir):
        return {"status": "completed", "message": "No research reports directory found. The archive is empty."}
        
    try:
        # Initialize or retrieve the RAG engine from the state to leverage caching
        if thread_safe_state is not None:
            if "rag_engine" not in thread_safe_state:
                thread_safe_state["rag_engine"] = TfIdfRetriever(data_dir=reports_dir)
            retriever = thread_safe_state["rag_engine"]
        else:
            retriever = TfIdfRetriever(data_dir=reports_dir)
            
        results = retriever.search(query=query, top_k=top_k)
        
        if not results:
            return {"status": "completed", "message": "No relevant information found in the research archive for this query."}
            
        formatted_results = []
        for i, res in enumerate(results):
            formatted_results.append(f"--- Result {i+1} (Source: {res['source']}, Relevance: {res['score']}) ---\n{res['text']}")
            
        combined_text = "\n\n".join(formatted_results)
        
        return {
            "status": "completed",
            "message": f"Found {len(results)} relevant chunks of past research.",
            "research_context": combined_text
        }
        
    except Exception as e:
        log(f"[Tool Error] 'query_research_archive' failed: {traceback.format_exc()}")
        return {"status": "error", "message": f"RAG query failed: {e}"}

def run_web_research(query: str, max_sites: int = 5, google_search_tool: Optional[Callable] = None, thread_safe_state: Dict[str, Any] = None) -> Dict[str, Any]:
    # Triggers a multi-agent team to research a query on the web.
    from agent import WebResearchCollaborator
    from ddgs import DDGS

    log(f"[Tool Trigger] Initiating 'run_web_research' for query: {query}")

    if not google_search_tool:
        log("[Web Research Team] 'google_search_tool' not injected. Falling back to local DDGS.")
        def _local_ddgs_search(queries, max_results_per_query=3):
            all_results = []
            with DDGS(timeout=20) as ddgs:
                for q in queries:
                    try:
                        results = ddgs.text(q, max_results=max_results_per_query)
                        if results:
                            for r in results:
                                all_results.append({"source_title": r.get("title", "No Title"), "url": r.get("href", ""), "snippet": r.get("body", "")})
                    except Exception as e:
                        log(f"[Search Tool ERROR] Failed to search for query '{q}': {e}")
            return all_results
        google_search_tool = _local_ddgs_search

    if thread_safe_state is None or "agent" not in thread_safe_state:
        return {"status": "error", "message": "Main agent is not initialized in thread_safe_state."}

    main_agent = thread_safe_state["agent"]

    try:
        collaborator = WebResearchCollaborator(main_agent, google_search_tool)
        result = collaborator.run_collaboration(query=query, max_sites=max_sites)

        # --- Phase-3 -> Theorist feed-forward (web report) ---
        # The Theorist never reliably retrieves a freshly-written report (RAG is lexical +
        # cached). Push a compact summary into one-shot state the Knowledge Scout injects
        # directly next cycle, and invalidate the RAG cache so the full report is also
        # retrievable on demand (rag_engine re-indexes the new file on next query).
        if isinstance(result, dict) and result.get("status") == "completed":
            report_text = str(result.get("report", ""))
            thread_safe_state["last_web_report"] = {
                "query": query,
                "summary": report_text[:800],
                "filepath": result.get("filepath"),
                "cycle": thread_safe_state.get("global_step_counter", 0),
            }
            thread_safe_state.pop("rag_engine", None)  # force re-index incl. the new report
        return result

    except Exception as e:
        log(f"[Tool Error] 'run_web_research' failed: {traceback.format_exc()}")
        return {"status": "error", "message": f"Web research collaboration failed: {e}"}


def generate_creative_hypotheses(query: str, thread_safe_state: Dict[str, Any], auto_build: bool = False) -> Dict[str, Any]:
    # Triggers a multi-agent "creative" team to brainstorm and select a new, data-driven hypothesis.
    # The chosen hypothesis is made ACTIONABLE: architectural winners are routed straight to the
    # Innovation Team (when auto_build=True), and every winner is fed forward to the next Theorist
    # cycle via thread_safe_state["last_creative_hypothesis"] — otherwise the result is dead text.
    from agent import CreativeHypothesisCollaborator

    log(f"[Tool Trigger] Initiating 'generate_creative_hypotheses' for query: {query}")

    if thread_safe_state is None or "agent" not in thread_safe_state:
        return {"status": "error", "message": "Main agent is not initialized in thread_safe_state."}

    main_agent = thread_safe_state["agent"]

    try:
        current_model_filter = thread_safe_state.get("selected_model")
        log(f"Gathering context for Creative Team, filtering for: '{current_model_filter}'")

        correlation_data = get_correlation_matrix(architecture_filter=current_model_filter, thread_safe_state=thread_safe_state)
        best_worst_data = analyze_best_vs_worst_trials(architecture_filter=current_model_filter, thread_safe_state=thread_safe_state)

        try:
            from main import convert_numpy_types
        except ImportError:
            convert_numpy_types = None

        def _safe_dumps(obj):
            if convert_numpy_types is not None:
                return json.dumps(convert_numpy_types(obj))
            return json.dumps(obj, default=str)

        current_data_context = (
            f"Current Data Analysis for '{current_model_filter}':\n"
            f"- Correlations: {_safe_dumps(correlation_data.get('correlation', 'N/A'))}\n"
            f"- Best/Worst Comparison: {_safe_dumps(best_worst_data)}"
        )

        strategic_log_data = thread_safe_state.get("strategic_log", [])

        collaborator = CreativeHypothesisCollaborator(main_agent)
        result = collaborator.run_collaboration(
            initial_query=query, current_data_context=current_data_context, strategic_log=strategic_log_data
        )

        # --- Make the chosen hypothesis actionable (otherwise it is dead text) ---
        if isinstance(result, dict) and result.get("status") == "completed":
            action_type = (result.get("action_type") or "unparsed").lower()
            directive = (result.get("directive") or "").strip()
            selected_model = thread_safe_state.get("selected_model", "")

            built = False
            # Route 1 (opt-in via auto_build): architectural winner -> Innovation Team directly,
            # mutating the current custom architecture. Self-contained; no Theorist dependency.
            if (auto_build and action_type == "architecture_upgrade" and directive
                    and isinstance(selected_model, str) and selected_model.endswith(".py")):
                log(f"[Creative Team] auto_build: routing architectural hypothesis to "
                    f"propose_architecture_upgrade (base='{selected_model}').")
                upgrade = propose_architecture_upgrade(
                    agent=main_agent, thread_safe_state=thread_safe_state,
                    base_file=selected_model, directive=directive,
                )
                result["architecture_upgrade_result"] = upgrade
                built = isinstance(upgrade, dict) and upgrade.get("status") == "completed"

            # Route 2 (feed-forward): if we did NOT already build, hand the directive to the next
            # Theorist cycle (one-shot). When built, the architecture registry already feeds the
            # new arch forward, so storing it again would cause a redundant rebuild proposal.
            if not built:
                thread_safe_state["last_creative_hypothesis"] = {
                    "action_type": action_type,
                    "directive": directive,
                    "justification": str(result.get("chosen_hypothesis_and_justification", ""))[:1500],
                    "cycle": thread_safe_state.get("global_step_counter", 0),
                }

        return result

    except Exception as e:
        log(f"[Tool Error] 'generate_creative_hypotheses' failed: {traceback.format_exc()}")
        return {"status": "error", "message": f"Creative collaboration failed: {e}"}


def run_architecture_comparison(
    architecture_A: str, architecture_B: str, query: str, execute_tool_func: Optional[Callable] = None, thread_safe_state: Dict[str, Any] = None
) -> Dict[str, Any]:
    # Triggers a multi-agent team to perform a deep comparison between two architectures.
    from agent import ArchitectureComparisonCollaborator

    log(f"[Tool Trigger] Initiating 'run_architecture_comparison' for {architecture_A} vs {architecture_B}")

    if not execute_tool_func:
        return {"status": "error", "message": "Tool executor function is not available or not injected."}

    if thread_safe_state is None or "agent" not in thread_safe_state:
        return {"status": "error", "message": "Main agent is not initialized in thread_safe_state."}

    main_agent = thread_safe_state["agent"]

    try:
        collaborator = ArchitectureComparisonCollaborator(main_agent, execute_tool_func)
        result = collaborator.run_collaboration(arch_A=architecture_A, arch_B=architecture_B, query=query)
        return result

    except Exception as e:
        log(f"[Tool Error] 'run_architecture_comparison' failed: {traceback.format_exc()}")
        return {"status": "error", "message": f"Architecture comparison failed: {e}"}


def run_strategic_graph_evaluation(execute_tool_func: Optional[Callable] = None, thread_safe_state: Dict[str, Any] = None, strategic_intent: Optional[str] = None) -> Dict[str, Any]:
    # Triggers a multi-agent team to evaluate the strategic graph and provide data-driven advice.
    from agent import StrategicGraphEvaluatorCollaborator
    import asyncio

    log(f"[Tool Trigger] Initiating 'run_strategic_graph_evaluation' (Intent: {strategic_intent})")

    if not execute_tool_func:
        return {"status": "error", "message": "Tool executor function is not available or not injected."}

    if thread_safe_state is None or "agent" not in thread_safe_state:
        return {"status": "error", "message": "Main agent is not initialized in thread_safe_state."}

    main_agent = thread_safe_state["agent"]

    try:
        collaborator = StrategicGraphEvaluatorCollaborator(main_agent, execute_tool_func)
        result = asyncio.run(collaborator.run_collaboration(thread_safe_state=thread_safe_state, strategic_intent=strategic_intent))
        return result

    except Exception as e:
        log(f"[Tool Error] 'run_strategic_graph_evaluation' failed: {traceback.format_exc()}")
        return {"status": "error", "message": f"Strategic graph evaluation failed: {e}"}


def get_strategic_graph(thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:
    # Returns a JSON representation of the strategic graph.
    log("Retrieving strategic graph...")
    if "strategic_graph_memory" in thread_safe_state:
        return thread_safe_state["strategic_graph_memory"].get_strategic_graph()
    else:
        return {"status": "error", "message": "Strategic graph memory not initialized in thread_safe_state."}



# ============================================================================
# NEW HYBRID GRAPH INTELLIGENCE TOOLS
# ============================================================================

def get_strategic_path(
    current_state_vector: Optional[List[float]] = None,
    goal_description: Optional[str] = None,
    max_path_length: int = 10,
    thread_safe_state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Finds the optimal path from current state to a high-value goal state using A* algorithm.
    Returns compressed path with actions and expected impacts (~500 tokens).
    
    Args:
        current_state_vector: Optional custom state. If None, uses current app state.
        goal_description: Optional goal description (not used yet, reserved for future)
        max_path_length: Maximum number of steps in path
        thread_safe_state: Application state
    
    Returns:
        Dict with 'status', 'path' (list of actions), and 'expected_improvement'
    """
    from rl_utils import encode_state
    
    if "strategic_graph_memory" not in thread_safe_state:
        return {"status": "error", "message": "Strategic graph memory not initialized."}
    
    graph_memory = thread_safe_state["strategic_graph_memory"]
    
    # Encode current state if not provided
    if current_state_vector is None:
        state_encoded = encode_state(thread_safe_state)
        current_state_vector = state_encoded.flatten().tolist()
    
    # Get current architecture
    current_arch = thread_safe_state.get("selected_model", "unknown")
    
    try:
        path = graph_memory.find_path(
            state_vector=current_state_vector,
            architecture=current_arch,
            goal_states=None,  # Auto-discover high-value states
            max_length=max_path_length
        )
        
        if not path:
            return {
                "status": "completed",
                "message": "No viable path found. Graph may be too sparse or no high-value states exist.",
                "path": [],
                "expected_improvement": 0.0
            }
        
        # Calculate total expected improvement
        total_q = sum(step["q_value"] for step in path)
        
        # Format path for readability
        formatted_path = []
        for i, step in enumerate(path):
            formatted_path.append({
                "step": i + 1,
                "action": step["action"],
                "expected_gain": round(step["q_value"], 4),
                "confidence": "high" if step["visits"] > 5 else "medium" if step["visits"] > 2 else "low",
                "visits": step["visits"],
                "arguments": step["full_arguments"]
            })
        
        return {
            "status": "completed",
            "path": formatted_path,
            "expected_improvement": round(total_q, 4),
            "path_length": len(path),
            "recommendation": f"Follow this {len(path)}-step path for an estimated +{total_q:.1%} improvement."
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Pathfinding failed: {str(e)}"}


def get_local_transitions(
    current_state_vector: Optional[List[float]] = None,
    top_k: int = 5,
    thread_safe_state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Returns the top K most promising actions from the current state based on Q-values.
    Provides compressed local context for expert LLM analysis (~300 tokens).
    
    Args:
        current_state_vector: Optional custom state. If None, uses current app state.
        top_k: Number of top transitions to return (default 5)
        thread_safe_state: Application state
        
    Returns:
        Dict with 'status' and 'transitions' list
    """
    from rl_utils import encode_state
    
    if "strategic_graph_memory" not in thread_safe_state:
        return {"status": "error", "message": "Strategic graph memory not initialized."}
    
    graph_memory = thread_safe_state["strategic_graph_memory"]
    
    # Encode current state if not provided
    if current_state_vector is None:
        state_encoded = encode_state(thread_safe_state)
        current_state_vector = state_encoded.flatten().tolist()
    
    # Get current architecture
    current_arch = thread_safe_state.get("selected_model", "unknown")
    
    try:
        transitions = graph_memory.get_available_transitions(
            state_vector=current_state_vector,
            architecture=current_arch
        )
        
        if not transitions:
            return {
                "status": "completed",
                "message": "No transitions found from current state. Graph may be empty or no matching anchors.",
                "transitions": []
            }
        
        # Sort by Q-value and take top K
        transitions.sort(key=lambda x: x.get("q_vector", [0])[0] if x.get("q_vector") else 0, reverse=True)
        top_transitions = transitions[:top_k]
        
        # Format for readability
        formatted = []
        for i, trans in enumerate(top_transitions):
            q_vec = trans.get("q_vector", trans.get("impact_vector", []))
            q_value = q_vec[0] if q_vec and len(q_vec) > 0 else 0
            
            formatted.append({
                "rank": i + 1,
                "action": trans["action"],
                "q_value": round(q_value, 4),
                "visits": trans["visits"],
                "confidence": "high" if trans["visits"] > 5 else "medium" if trans["visits"] > 2 else "low",
                "expected_impact": {
                    "accuracy": round(q_vec[0], 4) if q_vec and len(q_vec) > 0 else 0,
                    "stability": round(q_vec[1], 4) if q_vec and len(q_vec) > 1 else 0,
                },
                "arguments": trans.get("full_arguments", {})
            })
        
        return {
            "status": "completed",
            "transitions": formatted,
            "total_available": len(transitions),
            "showing_top": len(formatted)
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Failed to get transitions: {str(e)}"}


def get_hybrid_advice(
    strategic_intent: Optional[str] = None,
    thread_safe_state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Orchestrates pathfinding + LLM expert analysis via HybridGraphAdvisorCollaborator.
    Combines algorithmic A* pathfinding with Graph Strategist LLM synthesis (~1k tokens).

    Args:
        strategic_intent: Optional intent hint passed to the collaborator
        thread_safe_state: Application state (must contain 'agent')

    Returns:
        Dict with 'status', 'report' (Graph Strategist text), 'algorithmic_path'
    """
    if thread_safe_state is None or "agent" not in thread_safe_state:
        return {"status": "error", "message": "No agent in thread_safe_state."}

    try:
        import asyncio
        from agent import HybridGraphAdvisorCollaborator

        agent = thread_safe_state["agent"]
        collaborator = HybridGraphAdvisorCollaborator(main_agent=agent, execute_tool_func=None)

        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            result = loop.run_until_complete(
                collaborator.run_collaboration(thread_safe_state=thread_safe_state)
            )
        else:
            result = asyncio.run(
                collaborator.run_collaboration(thread_safe_state=thread_safe_state)
            )

        result["strategic_intent"] = strategic_intent or "balanced"
        return result

    except Exception as e:
        log(f"[get_hybrid_advice] HybridGraphAdvisorCollaborator failed: {e}. Falling back to stub.")
        # Graceful fallback: return path-only result without LLM synthesis
        path_result = get_strategic_path(thread_safe_state=thread_safe_state)
        return {
            "status": "completed",
            "report": path_result.get("recommendation", "No path available."),
            "algorithmic_path": path_result,
            "strategic_intent": strategic_intent or "balanced",
            "fallback": True,
        }

def generate_scientific_report(
    dataset: str, architecture: str, execute_tool_func: Optional[Callable] = None, thread_safe_state: Dict[str, Any] = None
) -> Dict[str, Any]:
    # Triggers a multi-agent team to generate a scientific report from the experiment data.
    from agent import ScientificReportGeneratorCollaborator
    import asyncio

    log(f"[Tool Trigger] Initiating 'generate_scientific_report'")

    if not execute_tool_func:
        return {"status": "error", "message": "Tool executor function is not available or not injected."}

    if thread_safe_state is None or "agent" not in thread_safe_state:
        return {"status": "error", "message": "Main agent is not initialized in thread_safe_state."}

    main_agent = thread_safe_state["agent"]
    
    # --- AUTO-RETRIEVE DATA FROM STATE ---
    # Prioritize 'report_data' (from file/folder loading) over current session history
    logs = thread_safe_state.get("report_data") or thread_safe_state.get("conversation_history", [])
    memory = thread_safe_state.get("memory") # Object or dict
    results_log = thread_safe_state.get("results_log", [])
    strategic_log = thread_safe_state.get("strategic_log", [])
    # -------------------------------------

    # Retrieve chunk_token_limit from config (default 50000)
    cfg = thread_safe_state.get("cfg")
    chunk_token_limit = 50000
    if cfg and hasattr(cfg, "execution"):
        chunk_token_limit = getattr(cfg.execution, "chunk_token_limit", 50000)
    elif cfg and isinstance(cfg, dict) and "execution" in cfg:
        chunk_token_limit = cfg["execution"].get("chunk_token_limit", 50000)

    try:
        collaborator = ScientificReportGeneratorCollaborator(main_agent, execute_tool_func, chunk_token_limit=chunk_token_limit)
        # Pass all retrieved data to the collaborator
        result = asyncio.run(collaborator.run_collaboration(dataset, architecture, logs, memory, results_log, strategic_log))
        return result

    except Exception as e:
        log(f"[Tool Error] 'generate_scientific_report' failed: {traceback.format_exc()}")
        return {"status": "error", "message": f"Scientific report generation failed: {e}"}


def generate_finetuning_dataset(
    execute_tool_func: Optional[Callable] = None, thread_safe_state: Dict[str, Any] = None
) -> Dict[str, Any]:
    # Triggers a multi-agent team to generate a finetuning dataset from the experiment data.
    from agent import FinetuningDatasetGeneratorCollaborator
    import asyncio

    log(f"[Tool Trigger] Initiating 'generate_finetuning_dataset'")

    if not execute_tool_func:
        return {"status": "error", "message": "Tool executor function is not available or not injected."}

    if thread_safe_state is None or "agent" not in thread_safe_state:
        return {"status": "error", "message": "Main agent is not initialized in thread_safe_state."}

    main_agent = thread_safe_state["agent"]

    # --- AUTO-RETRIEVE DATA FROM STATE ---
    logs = thread_safe_state.get("conversation_history", [])
    memory = thread_safe_state.get("memory")
    results_log = thread_safe_state.get("results_log", [])
    strategic_log = thread_safe_state.get("strategic_log", [])
    # -------------------------------------

    try:
        collaborator = FinetuningDatasetGeneratorCollaborator(main_agent, execute_tool_func)
        result = asyncio.run(collaborator.run_collaboration(logs, memory, results_log, strategic_log))
        return result

    except Exception as e:
        log(f"[Tool Error] 'generate_finetuning_dataset' failed: {traceback.format_exc()}")
        return {"status": "error", "message": f"Finetuning dataset generation failed: {e}"}


# In tools.py


def refresh_chronicle(thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:
    """Regenerate the technical Research Chronicle (deterministic metrics table + LLM-condensed
    insights) into workspace/research_chronicle.md and flag it for delivery to the Theorist on
    the NEXT cycle. Pull-based: the Theorist requests this when it needs the long-horizon
    technical history; the refreshed chronicle is injected into its next planning prompt."""
    try:
        from summarizer_ledger import build_research_chronicle
        chronicle = build_research_chronicle(thread_safe_state)
        if not chronicle:
            return {"status": "completed", "message": "No history yet — chronicle not generated."}
        thread_safe_state["chronicle_deliver_next"] = True
        return {
            "status": "completed",
            "message": "Research Chronicle refreshed; it will be injected into the Theorist on the next cycle.",
            "words": len(chronicle.split()),
        }
    except Exception as e:
        return {"status": "error", "message": f"Chronicle refresh failed: {e}"}


def read_file_from_architectures(filename: str, thread_safe_state: Dict[str, Any] = None) -> Dict[str, Any]:
    # Reads the content of a file from the custom architectures directory.
    # This is used to read both .py code files and .md changelog files.
    if ".." in filename or "/" in filename:
        return {"status": "error", "message": "Invalid filename. Do not use paths."}

    filepath = os.path.join(PERSISTENT_PATHS["custom_architectures_dir"], filename)

    if not os.path.exists(filepath):
        log(f"[Tool: read_file] File not found: {filename}")
        return {"status": "error", "message": "File not found.", "content": ""}

    try:
        with open(filepath, "r") as f:
            content = f.read()
        log(f"[Tool: read_file] Successfully read content from {filename}.")
        # Phase-3 -> Theorist feed-forward: when the Decider inspects a custom .py
        # architecture OTHER than the currently-selected one, record a compact insight.
        # The Theorist's Knowledge Scout otherwise only sees the selected arch's code,
        # so anything the Decider explores is lost next cycle.
        if (thread_safe_state is not None and filename.endswith(".py")
                and filename != thread_safe_state.get("selected_model")):
            thread_safe_state["last_arch_insight"] = {
                "filename": filename,
                "snippet": "\n".join(content.splitlines()[:40]),
                "cycle": thread_safe_state.get("global_step_counter", 0),
            }
        return {"status": "completed", "content": content}
    except Exception as e:
        log(f"[Tool: read_file] Error reading file {filename}: {e}")
        return {"status": "error", "message": f"Error reading file: {e}", "content": ""}


def append_to_architecture_changelog(py_filename: str, log_entry: str) -> Dict[str, Any]:
    # Appends a log entry to a .md file associated with a .py architecture file.
    # It automatically creates the .md file if it doesn't exist.
    if not py_filename.endswith(".py"):
        return {"status": "error", "message": "Filename must be the .py architecture file."}

    md_filename = py_filename.replace(".py", ".md")

    if ".." in md_filename or "/" in md_filename:
        return {"status": "error", "message": "Invalid filename."}

    filepath = os.path.join(PERSISTENT_PATHS["custom_architectures_dir"], md_filename)

    try:
        # Append mode 'a' will create the file if it doesn't exist
        with open(filepath, "a") as f:
            f.write(log_entry + "\n")
        log(f"[Tool: changelog] Appended entry to {md_filename}.")
        return {"status": "completed", "message": f"Changelog updated for {md_filename}."}
    except Exception as e:
        log(f"[Tool: changelog] Error appending to {md_filename}: {e}")
        return {"status": "error", "message": f"Error writing to changelog: {e}"}

def get_all_available_architectures() -> list:
    """
    Scans MODEL_BUILDERS and the custom_architectures directory to find all models.
    """
    # 1. Get base models from tools.py
    base_models = list(MODEL_BUILDERS.keys())

    # 2. Get custom .py models from the architectures directory
    custom_models = []

    # Use the correct path from state_manager.py
    arch_dir = PERSISTENT_PATHS.get("custom_architectures_dir")

    if arch_dir and os.path.exists(arch_dir):
        try:
            for f in os.listdir(arch_dir):
                # Ensure it's a python file and not a changelog (.md) or cache
                if f.endswith(".py"):
                    custom_models.append(f)
        except Exception as e:
            log(f"[APP WARNING] Could not scan custom_architectures_dir: {e}")
    else:
        log("[APP WARNING] custom_architectures_dir not found or not defined in PERSISTENT_PATHS.")

    # 3. Combine and de-duplicate, preserving order
    all_models = list(dict.fromkeys(base_models + custom_models))
    return all_models




def train_final_ensemble_models(base_test_dataset_id: str, thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trains final ensemble models on the best architectures using a dedicated worker process.
    """
    log("Delegating final ensemble training to a dedicated worker...")
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        result_path = tmp.name
    
    worker_script_path = os.path.join(PERSISTENT_PATHS["scripts_dir"], "ensemble_training_worker.py")
    
    command = [
        "python",
        worker_script_path,
        "--base_test_dataset_id",
        base_test_dataset_id,
        "--output_path",
        result_path,
    ]
    
    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=1800)  # 30 minutes timeout
        with open(result_path, "r") as f:
            result = json.load(f)
    except Exception as e:
        result = {
            "status": "error",
            "message": f"Ensemble training worker failed: {traceback.format_exc()}",
        }
    finally:
        if os.path.exists(result_path):
            os.remove(result_path)
            
    return result




def run_supernet_search(
    dataset_id: str,
    max_epochs: int = 100,
    population_size: int = 50,
    evolution_generations: int = 20,
    representation_type: Optional[str] = None,
    thread_safe_state: Dict[str, Any] = None,
    gpu_id: Optional[int] = None, # <--- AGGIUNTO QUI
    save_model: bool = False,
    **kwargs # Robustness: Handle hallucinated args
) -> Dict[str, Any]:
    """
    Runs a One-Shot NAS supernet search to find an optimal architecture.
    
    Args:
        dataset_id: The ID of the dataset manifest.
        max_epochs: Number of search epochs (default 100).
        population_size: Evolution population size (default 50).
        evolution_generations: Evolution generations (default 20).
        representation_type: The data representation (e.g., '1D_CNN', '2D_IMAGE').
        layers: (Optional) Number of Supernet layers (default 8). Reduce to 4-6 if OOM occurs.
        init_channels: (Optional) Initial channel count (default 16). Reduce to 8 if OOM occurs.
    """
    if thread_safe_state is None:
        return {"status": "error", "message": "Thread safe state not provided."}

    # --- CONFIG TOGGLE (SUPERNET DEACTIVATION) ---
    cfg = thread_safe_state.get("cfg", {})
    execution_cfg = cfg.get("execution", {})
    # Fallback is False per user request: if not explicitly enabled, it's disabled.
    enable_supernet = execution_cfg.get("enable_supernet", False)
    
    if not enable_supernet:
        log("[Supernet] Blocked: Supernet tool is disabled in config.yaml.")
        return {
            "status": "error",
            "message": "Supernet Neural Architecture Search is currently disabled by configuration (execution.enable_supernet: false). You MUST try standard architectures, manual parameter tuning, or use propose_intelligent_architecture to proceed."
        }
    # ---------------------------------------------

    # Robustness: Map common hallucinations
    if "max_trials" in kwargs:
         log(f"[Supernet] Warning: 'max_trials' is not a valid argument. Using it as 'max_epochs' ({kwargs['max_trials']}).")
         max_epochs = kwargs["max_trials"]

    log(f"[Supernet] Starting search on {dataset_id} (epochs={max_epochs})")
    
    # Cache check moved after representation inference

    # Resolve Data Path
    try:
        metadata = _load_dataset_metadata(thread_safe_state)
        if dataset_id not in metadata:
             return {"status": "error", "message": f"Dataset '{dataset_id}' not found."}
        
        manifest_data = metadata[dataset_id]
        raw_metadata_path = manifest_data["path"]
        
        # Smart Path Resolution: Check if metadata path is valid, else fallback to local
        if os.path.exists(raw_metadata_path):
             data_path_root = raw_metadata_path
        else:
             # Construct expected local path standard
             local_path = os.path.join(PERSISTENT_PATHS["generated_datasets_dir"], dataset_id)
             if os.path.exists(local_path):
                 log(f"[Supernet] Resolved local data path: {local_path} (Metadata path '{raw_metadata_path}' not found)")
                 data_path_root = local_path
             else:
                 # Fallback to metadata path so it fails with a clear error later or works if network path
                 data_path_root = raw_metadata_path
        
        # --- AUTO-INFERENCE OF REPRESENTATION TYPE ---
        if representation_type is None:
            # 0. NEW: Check if selected_model matches a valid representation in the manifest
            current_model = thread_safe_state.get("selected_model")
            if current_model and current_model in manifest_data.get("generated_representations", {}):
                 log(f"[Supernet] Auto-selected representation from current model focus: {current_model}")
                 representation_type = current_model

            # 1. Try to get the recommended architecture from the manifest
            if not representation_type:
                representation_type = manifest_data.get("recommended_architecture")
            
            # 2. If that fails, try to infer from the 'generated_representations' keys
            if not representation_type:
                generated = manifest_data.get("generated_representations", {})
                if generated:
                    # Pick the first one (usually there's only one relevant one)
                    representation_type = list(generated.keys())[0]
                    log(f"[Supernet] Auto-selected representation: {representation_type}")
            
            # 3. If still None, default to "2D_IMAGE" (legacy behavior) but warn
            if not representation_type:
                log("[Supernet] Warning: Could not infer representation type. Defaulting to '2D_IMAGE'.")
                representation_type = "2D_IMAGE"
            else:
                log(f"[Supernet] Inferred representation type: {representation_type}")
        # ---------------------------------------------

        # --- SUPERNET CACHING MECHANISM (MOVED) ---
        # Check if we already have a valid genotype for this configuration AND REPRESENTATION.
        try:
            cache_path = os.path.join(PERSISTENT_PATHS["processed_data_dir"], "supernet_cache.json")
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    supernet_cache = json.load(f)
                
                # Key based on dataset and THE NOW RESOLVED representation_type
                cache_key = f"{dataset_id}_{representation_type}" 
                if cache_key in supernet_cache:
                    cached_data = supernet_cache[cache_key]
                    log(f"[Supernet] ⚡ CACHE HIT: Found existing genotype for {cache_key}. Skipping search.")
                    return cached_data
        except Exception as e:
            log(f"[Supernet Warning] Cache check failed: {e}")
        # ------------------------------------------
        
        # VALIDATION
        VALID_REPRESENTATION_TYPES = [
            "1D_CNN", "2D_SPECTROGRAM", "2D_GAF", "2D_CWT_SCALOGRAM",
            "2D_GENERIC_IMAGE", "2D_IMAGE", "3D_VIDEO", "3D_GAF_VIDEO",
            "3D_DYNAMIC_GAF", "3D_DYNAMIC_CWT", "3D_WAVELET_CWT"
        ]
        if representation_type not in VALID_REPRESENTATION_TYPES:
             return {
                 "status": "error", 
                 "message": f"Invalid representation_type '{representation_type}'. Must be one of: {VALID_REPRESENTATION_TYPES}"
             }
        
        # Check if representation exists
        if representation_type not in manifest_data.get("generated_representations", {}):
             # Try to generate it? Or just error for now.
             # For supernet, we assume data is ready or we trigger generation.
             # Let's try to trigger generation if missing, similar to run_training_trial
             log(f"[Supernet] Data for '{representation_type}' missing. Attempting generation...")
             gen_result = generate_representation(dataset_id, representation_type, thread_safe_state)
             if gen_result.get("status") != "completed":
                 return {"status": "error", "message": f"Data generation failed: {gen_result.get('message')}"}
             
    except Exception as e:
        return {"status": "error", "message": f"Data resolution failed: {e}"}

    # Prepare command
    worker_script = os.path.join(PERSISTENT_PATHS["scripts_dir"], "training_worker.py")
    
    # Create temp file for results
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        result_path = tmp.name

    # We pass representation_type as experiment_name so worker finds {representation_type}_data.npy
    command = [
        "python", worker_script,
        "--mode", "supernet",
        "--dataset_id", dataset_id,
        "--epochs", str(max_epochs),
        "--output_path", result_path,
        "--data_path_root", data_path_root,
        "--experiment_name", representation_type, 
        "--params_json", "{}" 
    ]
    
    # NOTE: We do NOT pass --gpu_id as argument to avoid worker overriding CUDA_VISIBLE_DEVICES.
    # The environment variable set by _get_subprocess_env (line 2628) is the sole source of truth.
    # if gpu_id is not None:
    #     command.extend(["--gpu_id", str(gpu_id)])
    
# Add import at the top (assumed to be done manually or via this block if I can match the top)
# I will add the import inside the functions to avoid messing with top-level imports if I can't see them all, 
# but better to add it at the top if possible. 
# Since I can't easily see the top, I'll use a local import for safety in this tool call, 
# or just assume I can edit the top.
# Let's edit the functions directly.

    # --- CENTRALIZED GPU LOGIC (Agent assigns GPU) ---
    if gpu_id is not None:
        command.extend(["--gpu_id", str(gpu_id)])
        log(f"[Supernet] Using ASSIGNED orchestrator GPU {gpu_id}")
    else:
        log("[Supernet] No internal GPU ID assigned. Running strictly CPU or default.")

    success = False
    try:
        log(f"[Supernet] Executing: {' '.join(command)}")
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=86400)
        success = True # Assumed success if no exception, file check happens later
             
    except Exception as e:
        log(f"[Supernet] Single-GPU run failed: {e}")

    if not success:
         return {"status": "error", "message": "Supernet Search failed."}

    # Read result (from the successful run)
    try:
        with open(result_path, "r") as f:
            result = json.load(f)
    except Exception as e:
         return {"status": "error", "message": f"Failed to read result file after success: {e}"}
        
    if result.get("status") == "completed":
            genotype = result.get("genotype")
            # Support both keys for robustness
            accuracy = result.get("accuracy") or result.get("final_val_accuracy") or 0.0
            log(f"[Supernet] Search completed. Genotype found: {genotype}")
            
            # Save to results log (Search Phase)
            if "results_log" in thread_safe_state:
                arch_name_for_log = kwargs.get("experiment_name") or thread_safe_state.get("selected_model") or representation_type
                log_entry = {
                    "source": "supernet_search",
                    "architecture": arch_name_for_log,
                    "manifest_name": dataset_id,
                    "genotype": genotype,
                    "mean_accuracy": accuracy,
                    "std_accuracy": 0.0,
                    "status": "completed",
                    "params": {"max_epochs": max_epochs}
                }
                thread_safe_state["results_log"].append(log_entry)
                save_state(thread_safe_state)

            
            # --- AUTONOMOUS ROUTINE: CONSTRUCTION & OPTIMIZATION ---
            # As per "Coding Team" protocol: The tool completes the full workflow.
            
            routine_report = []
            routine_report.append(f"1. Supernet Search successful (Acc: {accuracy:.4f}).")
            
            # Step 1: Construct Model
            try:
                # Assuming construct_model_from_genotype matches the signature found in this file
                # We need a model name. 
                timestamp = int(time.time())
                model_name = f"{representation_type}_Genotype_{timestamp}"
                
                construction_result = construct_model_from_genotype(genotype, model_name, thread_safe_state)
                
                if construction_result.get("status") == "completed":
                    filename = construction_result.get("filepath") # wait, tool returns 'filepath' or 'filename'?
                    # Looking at construct_model_from_genotype implementation below:
                    # returns {"status": "completed", ..., "filepath": filepath}
                    filename = os.path.basename(construction_result.get("filepath"))
                    routine_report.append(f"2. Model Constructed: {filename}")
                    
                    # Step 2: Optuna Optimization
                    # We run a quick sweep to validate the new model
                    log(f"[Supernet Routine] Launching Optuna Sweep for {filename}...")
                    
                    custom_params = kwargs.get("optuna_hyperparameters")
                    
                    default_params = {
                        "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 1e-2},
                        "dropout_rate": {"type": "float", "min": 0.0, "max": 0.6},
                        "batch_size": {"type": "choice", "choices": [16, 32, 64, 128]},
                        # Expanded Search Space for Architecture
                        "init_channels": {"type": "choice", "choices": [16, 24, 32, 48, 64]},
                        "layers": {"type": "choice", "choices": [8, 10, 12, 14]},
                        "optimizer": {"type": "choice", "choices": ["adam", "sgd"]}
                    }
                    
                    # Use custom if provided, otherwise default
                    final_hyperparameters = custom_params if custom_params else default_params

                    sweep_result = run_strategic_optuna_sweep(
                        thread_safe_state=thread_safe_state,
                        n_trials=10, # Keep it light for the routine
                        manifest_name=dataset_id,
                        experiment_name=model_name, # Use model name as experiment name
                        custom_architecture_file=filename,
                        representation_type=representation_type,
                        optimization_focus="accuracy",
                        gpu_id=target_gpu_id, # Reuse the same GPU if assigned
                        hyperparameters=final_hyperparameters
                    )
                    
                    if sweep_result.get("status") == "completed":
                        # CRITICAL FIX: Handle new strategic_report structure from multi-objective Optuna
                        strategic_report = sweep_result.get("strategic_report", {})
                        if strategic_report:
                            best_val = strategic_report.get("sweep_summary", {}).get("best_value_achieved", 0.0)
                        else:
                            # Fallback for old structure or direct result
                            best_trial = sweep_result.get("best_trial", {})
                            best_val = best_trial.get("value", 0.0)
                        
                        routine_report.append(f"3. Optuna Sweep Completed. Best Acc: {best_val:.4f}")
                        
                        aggregated_result = {
                            "status": "completed",
                            "genotype": genotype,
                            "accuracy": accuracy,
                            "final_model_file": filename,
                            "optimization_result": sweep_result,
                            "routine_summary": "\n".join(routine_report)
                        }

                        # --- STEP 4: AUTONOMOUS HANDOFF (OPTIONAL) ---
                        # If the result is good but not perfect (e.g. < 0.95), and we have the agent,
                        # we autonomously call the Innovation Team to refine it.
                        TARGET_ACCURACY = 0.95
                        
                        # ROBUST AGENT RETRIEVAL
                        agent_instance = kwargs.get("agent")
                        if not agent_instance and thread_safe_state:
                            agent_instance = thread_safe_state.get("agent")
                        
                        log(f"[Supernet Routine] Checking Handoff Conditions: Best Val={best_val:.3f}, Target={TARGET_ACCURACY}, Agent Available={str(bool(agent_instance))}")

                        if best_val < TARGET_ACCURACY:
                            if agent_instance:
                                log(f"[Supernet Routine] ⚠️ Accuracy ({best_val:.3f}) < Target ({TARGET_ACCURACY}). Autonomous Handoff to Innovation Team triggered.")
                                routine_report.append("4. Autonomous Handoff: Triggering Innovation Team for refinement.")
                                
                                try:
                                    # We need to avoid infinite recursion if Innovation Team calls Supernet again.
                                    # But Innovation Team (run_innovation_cycle) defaults to Lite check or Logic.
                                    import traceback
                                    log("[Supernet Routine] Calling propose_intelligent_architecture...")
                                    
                                    refine_hypothesis = f"Refine the Supernet backbone '{filename}' (Acc: {best_val:.3f}). Improve architecture to reach >95% accuracy."
                                    
                                    refinement_result = propose_intelligent_architecture(
                                        agent=agent_instance, 
                                        thread_safe_state=thread_safe_state,
                                        hypothesis=refine_hypothesis,
                                        architecture_context=filename,
                                        # limit depth to prevent loops
                                        depth=kwargs.get("depth", 0) + 1 
                                    )
                                    
                                    aggregated_result["innovation_refinement"] = refinement_result
                                    if refinement_result.get("status") == "completed":
                                        # If refinement succeeded, we might want to point to THAT as the final result?
                                        # For now, we just attach it. The Agent will see the update in the logs/state.
                                        routine_report.append("4. Innovation Team Refinement Completed.")
                                    else:
                                        routine_report.append(f"4. Innovation Team Refinement Failed: {refinement_result.get('message')}")
    
                                except Exception as e:
                                    log(f"[Supernet Routine] Handoff failed: {e}")
                                    routine_report.append(f"4. Handoff Failed: {e}")
                            else:
                                log("[Supernet Routine] Handoff skipped: Agent instance not available.")
                                routine_report.append("4. Autonomous Handoff Skipped: Agent not available.")
                        else:
                            log(f"[Supernet Routine] Handoff skipped: Accuracy {best_val:.3f} meets target {TARGET_ACCURACY}.")
                            routine_report.append(f"4. Autonomous Handoff Skipped: Target accuracy met.")
                        
                        aggregated_result["routine_summary"] = "\n".join(routine_report)
                        # ---------------------------------------------

                        # --- CACHE WRITE ---
                        try:
                            cache_path = os.path.join(PERSISTENT_PATHS["processed_data_dir"], "supernet_cache.json")
                            current_cache = {}
                            if os.path.exists(cache_path):
                                with open(cache_path, "r") as f:
                                    current_cache = json.load(f)
                            
                            cache_key = f"{dataset_id}_{representation_type or 'auto'}"
                            current_cache[cache_key] = aggregated_result
                            
                            with open(cache_path, "w") as f:
                                json.dump(current_cache, f, indent=4)
                            log(f"[Supernet] ⚡ Caching successful result for {cache_key}")
                        except Exception as e:
                            log(f"[Supernet Warning] Failed to write cache: {e}")
                        # -------------------

                        return aggregated_result
                    else:
                        routine_report.append(f"3. Optuna Sweep Failed: {sweep_result.get('message')}")
                        return {"status": "completed", "genotype": genotype, "message": "Search & Construction success, but Optimization failed.", "details": "\n".join(routine_report)}

                else:
                    return {"status": "completed", "genotype": genotype, "message": f"Construction failed: {construction_result.get('message')}"}
                    
            except Exception as e:
                log(f"[Supernet Routine] Autonomous steps failed: {e}")
                return {"status": "completed", "genotype": genotype, "message": f"Search success, but routine crashed: {e}"}

            # ----------------------------------------------------


def construct_model_from_genotype(
    genotype: Dict[str, Any],
    model_name: str,
    thread_safe_state: Dict[str, Any]
) -> Dict[str, Any]:
    log(f"[Tool: construct_model] Generazione Modello Corretta '{model_name}'...")

    if not model_name.endswith(".py"):
        model_name += ".py"
        
    # Usiamo repr(genotype) invece di json.dumps per evitare il problema 'null' vs 'None'
    python_genotype = repr(genotype)

    code_content = f'''
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
    genotype = {python_genotype}
    
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
    if params.get(\"dropout_rate\", 0) > 0:
        out = layers.Dropout(params.get(\"dropout_rate\"))(out)
        
    outputs = layers.Dense(num_classes, activation=\"softmax\", kernel_regularizer=reg)(out)
    model = keras.Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params.get(\"learning_rate\", 1e-3)),
        loss=\"categorical_crossentropy\",
        metrics=[\"accuracy\"]
    )
    return model
'''
    filepath = os.path.join(PERSISTENT_PATHS["custom_architectures_dir"], model_name)
    with open(filepath, "w") as f:
        f.write(code_content)
    
    # IMPORTANTE: aggiorniamo il file memorizzato nell'Agente
    if thread_safe_state:
        thread_safe_state["latest_created_architecture"] = model_name
    return {"status": "completed", "filepath": filepath}

# --- ToDo List Tools ---

def create_todo_list(thread_safe_state: Dict[str, Any], tasks: list[str] = None, content: str = None) -> Dict[str, Any]:

    """

    Creates a new TODO.md file with a list of tasks.

    """
    
    # Handle the case where the agent provides 'content' instead of 'tasks'
    if tasks is None and content is not None:
        # Simple parsing of content string into a list of tasks
        # Assuming tasks are separated by newlines
        tasks = [line.strip().lstrip('- ').strip() for line in content.split('\n') if line.strip()]

    if tasks is None:
         return {"status": "error", "message": "Neither 'tasks' nor 'content' provided."}

    log("[TODO] Creating a new to-do list.")
    try:
        new_todos = [{"description": task, "status": "pending"} for task in tasks]
        update_todos(thread_safe_state, new_todos)
        return {"status": "completed", "message": "To-Do list created and saved."}
    except Exception as e:

        return {"status": "error", "message": f"Failed to create TODO.md file: {e}"}



def read_todo_list(thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:

    """

    Reads the content of the TODO.md file.

    """

    log("[TODO] Reading the to-do list.")
    try:
        todos = get_todos(thread_safe_state)
        if not todos:
            return {"status": "completed", "message": "To-Do list is empty."}

        formatted_todos = "# To-Do List\n\n"
        for i, todo in enumerate(todos):
            status_char = "[x]" if todo["status"] == "completed" else "[ ]"
            formatted_todos += f"- {status_char} {i+1}. {todo['description']}\n"
        return {"status": "completed", "content": formatted_todos}
    except Exception as e:
        return {"status": "error", "message": f"Failed to read to-do list: {e}"}

# --- Late Registration of Tools ---
# Register tools that are defined after the initial AVAILABLE_TOOLS dictionary
AVAILABLE_TOOLS["construct_model_from_genotype"] = construct_model_from_genotype



def update_todo_list(task_number: int, completed: bool, thread_safe_state: Dict[str, Any]) -> Dict[str, Any]:

    """

    Updates a task in the TODO.md file.

    """

    log(f"[TODO] Updating task {task_number} to completed={completed}.")
    try:
        todos = get_todos(thread_safe_state)
        if not todos:
            return {"status": "error", "message": "To-Do list is empty. No tasks to update."}

        if task_number < 1 or task_number > len(todos):
            return {"status": "error", "message": f"Invalid task number: {task_number}. There are {len(todos)} tasks."}

        # Adjust for 0-based indexing
        task_index = task_number - 1
        todos[task_index]["status"] = "completed" if completed else "pending"
        update_todos(thread_safe_state, todos) # Save the updated list

        return {"status": "completed", "message": f"Task {task_number} updated to {'completed' if completed else 'pending'}."}
    except Exception as e:
        return {"status": "error", "message": f"Failed to update to-do list: {e}"}


# --------------------------------------------------------------------------------
# MANUAL MOSAN TOOL
# --------------------------------------------------------------------------------

def consult_strategic_advisor(thread_safe_state: Dict[str, Any], focus_area: str = "general") -> str:
    """
    Manually consults the Multi-Agent Strategic Advisor Team (MOSAN).
    Useful when you feel stuck or need a strategic "second opinion" outside the automatic loop.
    
    Args:
        focus_area: Optional context hint (e.g. "debugging", "novelty", "efficiency").
    """
    from cluster.tasks import run_planning_advisor_task
    import uuid
    
    # Create a snapshot for the worker
    snapshot = {
        "selected_model": thread_safe_state.get("selected_model"),
        "results_log": thread_safe_state.get("results_log", []),
        "todos": thread_safe_state.get("todos", []),
        "architecture_flags": thread_safe_state.get("architecture_flags", {}),
        "task_type": focus_area # Hint for the lens
    }
    
    run_id = f"manual_consult_{uuid.uuid4().hex[:6]}"
    log(f"[Tool] Manually consulting Strategic Advisor (RunID: {run_id})...")
    
    try:
        # Check if Celery is available or just run sync if in test/debug
        # We assume Celery is up if we are in agent loop.
        # But if we are local, apply_async might fail if broker not found.
        # Fallback to sync call if import fails? No, run_planning_advisor_task is a @task.
        
        async_result = run_planning_advisor_task.apply_async(args=[snapshot, run_id])
        advice = async_result.get(timeout=60) # Wait up to 60s
        
        if not advice:
            return "Strategic Advisor is currently offline or returned no advice."
            
        return f"**ADVISOR TEAM RESPONSE:**\n{advice}"
        
    except Exception as e:
        return f"Failed to consult advisor: {str(e)}"
        
# Late Registration
AVAILABLE_TOOLS["consult_strategic_advisor"] = consult_strategic_advisor


# ============================================================
# IMAGENET COMPETITION TOOLS
# These are thin dispatcher functions — heavy logic is in workers.
# They are fully isolated from the optimization/report pipeline.
# ============================================================

def run_imagenet_training(
    thread_safe_state: Dict[str, Any],
    backbone_depth: int = 152,
    batch_size_per_gpu: int = 256,
    base_lr: float = 0.1,
    warmup_epochs: int = 5,
    total_epochs: int = 90,
    mixed_precision: bool = True,
    xla: bool = True,
    bbox_loss_weight: float = 1.0,
) -> Dict[str, Any]:
    """
    Agent-facing tool: Dispatches an ILSVRC ImageNet training campaign via Celery.

    The agent should call this when it wants to start or continue a training run.
    Returns immediately (async) with a Celery task ID to monitor.

    Args:
        backbone_depth: ResNet depth — 50, 101, or 152. Agent ablates this.
        batch_size_per_gpu: Per-GPU batch size (global = × num_gpus).
        base_lr: Base learning rate (auto-scaled by linear LR rule).
        warmup_epochs: LR warmup epochs.
        total_epochs: Total training epochs.
        mixed_precision: Use bfloat16 mixed precision.
        xla: Enable XLA jit_compile on training step.
        bbox_loss_weight: Lambda for SmoothL1 bounding box loss.

    Returns:
        dict with "task_id" for monitoring, and config echo.
    """
    cfg = thread_safe_state.get("cfg")
    imagenet_cfg = cfg.get("imagenet", {}) if cfg else {}

    train_args = {
        "dataset_dir": imagenet_cfg.get("dataset_dir", "processed_data/imagenet/train"),
        "val_dir": imagenet_cfg.get("val_dir", "processed_data/imagenet/val"),
        "checkpoint_dir": imagenet_cfg.get("checkpoint_dir", "processed_data/imagenet/checkpoints"),
        "synset_map": imagenet_cfg.get("synset_map", "imagenet/synset_mapping.txt"),
        "backbone_depth": backbone_depth,
        "batch_size_per_gpu": batch_size_per_gpu,
        "base_lr": base_lr,
        "warmup_epochs": warmup_epochs,
        "total_epochs": total_epochs,
        "mixed_precision": mixed_precision,
        "xla": xla,
        "bbox_loss_weight": bbox_loss_weight,
        "mlflow_experiment": "ImageNet_Competition",
    }

    try:
        result = run_imagenet_training_task.apply_async(args=[train_args])
        log(f"[Tool] run_imagenet_training dispatched. Task ID: {result.id}")
        return {
            "status": "dispatched",
            "task_id": result.id,
            "message": (
                f"Training dispatched asynchronously. "
                f"ResNet-{backbone_depth}, {batch_size_per_gpu}×GPU batch. "
                f"Monitor in MLFlow under 'ImageNet_Competition'."
            ),
            "config": train_args,
        }
    except Exception as e:
        log(f"[Tool] run_imagenet_training FAILED: {e}")
        return {"status": "error", "message": str(e)}


def run_imagenet_inference(
    thread_safe_state: Dict[str, Any],
    checkpoint_path: Optional[str] = None,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Agent-facing tool: Runs inference on the ImageNet test set using a trained checkpoint.

    If no checkpoint_path is provided, the tool looks for the latest checkpoint in
    the directory specified in config.yaml (imagenet.checkpoint_dir).

    Returns:
        dict with task_id for monitoring, or immediate result if running synchronously.
    """
    cfg = thread_safe_state.get("cfg")
    imagenet_cfg = cfg.get("imagenet", {}) if cfg else {}
    checkpoint_dir = imagenet_cfg.get("checkpoint_dir", "processed_data/imagenet/checkpoints")

    # Auto-discover checkpoint if not specified
    if not checkpoint_path:
        from imagenet.competition_runner import _find_best_checkpoint
        checkpoint_path = _find_best_checkpoint(checkpoint_dir)
        if not checkpoint_path:
            return {
                "status": "error",
                "message": (
                    "No checkpoint found. Run 'run_imagenet_training' first and wait "
                    "for a checkpoint to be saved."
                ),
            }

    infer_args = {
        "checkpoint_path": checkpoint_path,
        "test_dir": imagenet_cfg.get("test_dir", "processed_data/imagenet/test"),
        "synset_map": imagenet_cfg.get("synset_map", "imagenet/synset_mapping.txt"),
        "output_path": "outputs/imagenet",
        "batch_size": batch_size,
    }

    try:
        result = run_imagenet_inference_task.apply_async(args=[infer_args])
        log(f"[Tool] run_imagenet_inference dispatched. Task ID: {result.id}")
        return {
            "status": "dispatched",
            "task_id": result.id,
            "checkpoint_path": checkpoint_path,
            "message": "Inference dispatched. Results will appear in outputs/imagenet/.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def evaluate_imagenet_predictions(
    predictions_json_path: str,
    ground_truth_json_path: str,
    thread_safe_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Agent-facing tool: Computes the ILSVRC error metric on a validation set.

    Allows the agent to evaluate prediction quality locally before submitting to Kaggle.

    Args:
        predictions_json_path: Path to JSON file with prediction dicts (same format as submission).
        ground_truth_json_path: Path to JSON file with ground truth in the same format.

    Returns:
        dict with error_rate, num_images, num_correct.
    """
    try:
        with open(predictions_json_path, "r") as f:
            predictions = json.load(f)
        with open(ground_truth_json_path, "r") as f:
            ground_truth = json.load(f)

        from imagenet.submission_generator import compute_competition_error
        metrics = compute_competition_error(predictions, ground_truth)
        log(f"[Tool] evaluate_imagenet_predictions: error_rate={metrics['error_rate']:.4f}")
        return {"status": "completed", **metrics}
    except FileNotFoundError as e:
        return {"status": "error", "message": f"File not found: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Evaluation failed: {e}"}


def generate_kaggle_submission(
    predictions_json_path: str,
    output_csv_path: str,
    thread_safe_state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Agent-facing tool: Converts raw predictions JSON to a Kaggle submission CSV.

    The agent can call this after 'run_imagenet_inference' to produce a submission file.

    Args:
        predictions_json_path: Path to raw predictions JSON (output of inference worker).
        output_csv_path: Where to write the Kaggle submission CSV.

    Returns:
        dict with submission_csv_path and row count.
    """
    try:
        with open(predictions_json_path, "r") as f:
            predictions = json.load(f)

        from imagenet.submission_generator import SubmissionGenerator
        gen = SubmissionGenerator()
        csv_path = gen.write_csv(predictions, output_csv_path)

        log(f"[Tool] generate_kaggle_submission: {len(predictions)} rows → {csv_path}")
        return {
            "status": "completed",
            "submission_csv_path": csv_path,
            "num_rows": len(predictions),
            "message": f"Kaggle submission CSV written to {csv_path}. Upload to: https://www.kaggle.com/competitions/imagenet-object-localization-challenge",
        }
    except FileNotFoundError as e:
        return {"status": "error", "message": f"Predictions file not found: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Submission generation failed: {e}"}
