import csv
import json
import os
from datetime import datetime

from ui_logger import log

# --- 1. PERCORSI DINAMICI E CENTRALIZZATI (CORRETTI) ---

# Trova il percorso assoluto della cartella principale del progetto
# La riga seguente è stata corretta per puntare alla directory corretta.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# IMPORTANT FOR CLUSTER DEPLOYMENT:
# In a multi-node cluster environment, these paths should point to a shared
# file system (e.g., an NFS mount) that is accessible from all worker nodes.
# This ensures that all nodes read from and write to the same data, models,
# and logs.
# ==============================================================================
PERSISTENT_PATHS = {
    # Directory principali
    "processed_data_dir": os.path.join(PROJECT_ROOT, "processed_data"),
    "logs_dir": os.path.join(PROJECT_ROOT, "processed_data", "logs"),
    "raw_data_dir": os.path.join(PROJECT_ROOT, "processed_data", "raw_data"),
    "long_term_memory_dir": os.path.join(PROJECT_ROOT, "processed_data", "long_term_memory"),
    "rl_models_dir": os.path.join(PROJECT_ROOT, "processed_data", "rl_models"),
    "generated_datasets_dir": os.path.join(PROJECT_ROOT, "processed_data", "generated_datasets"),
    "ensemble_models_dir": os.path.join(PROJECT_ROOT, "processed_data", "ensemble_models"),
    "custom_architectures_dir": os.path.join(PROJECT_ROOT, "processed_data", "custom_architectures"),
    "strategic_memory_dir": os.path.join(PROJECT_ROOT, "processed_data", "strategic_memory"),
    "research_reports_dir": os.path.join(PROJECT_ROOT, "processed_data", "research_reports"),
    "models_dir": os.path.join(PROJECT_ROOT, "processed_data", "models"),
    "trials_output_dir": os.path.join(PROJECT_ROOT, "processed_data", "trials"),
    # Percorsi degli script worker
    "scripts_dir": os.path.join(PROJECT_ROOT, "processed_data", "scripts_worker"),
    # File specifici
    "results_log": os.path.join(PROJECT_ROOT, "processed_data", "logs", "results_log.json"),
    "finetune_results_log": os.path.join(PROJECT_ROOT, "processed_data", "logs", "finetune_results_log.json"),
    "conversation_history": os.path.join(PROJECT_ROOT, "processed_data", "logs", "conversation_history.json"),
    "datasets_metadata": os.path.join(PROJECT_ROOT, "processed_data", "generated_datasets", "datasets_metadata.json"),
    "strategic_log": os.path.join(PROJECT_ROOT, "processed_data", "logs", "strategic_log.json"),
    "architecture_flags": os.path.join(PROJECT_ROOT, "processed_data", "logs", "architecture_flags.json"),
    "todos_file": os.path.join(PROJECT_ROOT, "processed_data", "todos.json"),
    "session_metadata": os.path.join(PROJECT_ROOT, "processed_data", "logs", "session_metadata.json"),
    "detailed_traces_dir": os.path.join(PROJECT_ROOT, "processed_data", "logs", "detailed_traces"),
    "mlruns_dir": os.path.join(PROJECT_ROOT, "processed_data", "mlruns"),
    "gpu_locks_dir": os.path.join(PROJECT_ROOT, "processed_data", "locks"),
}


def _ensure_dirs_exist():
    """Ensures that all necessary directories exist before read/write."""
    for path_key, path_value in PERSISTENT_PATHS.items():
        if path_key.endswith("_dir"):
            os.makedirs(path_value, exist_ok=True)


def save_state(session_state):
    """Saves the session state (results and conversation) to disk."""
    try:
        log("[State Manager] Attempting to save state...")
        _ensure_dirs_exist()

        with open(PERSISTENT_PATHS["results_log"], "w") as f:
            json.dump(session_state.get("results_log", []), f, indent=2)

        with open(PERSISTENT_PATHS["conversation_history"], "w") as f:
            json.dump(session_state.get("conversation_history", []), f, indent=2)

        with open(PERSISTENT_PATHS["strategic_log"], "w") as f:
            json.dump(session_state.get("strategic_log", []), f, indent=2)

        with open(PERSISTENT_PATHS["architecture_flags"], "w") as f:
            json.dump(session_state.get("architecture_flags", {}), f, indent=2)

        with open(PERSISTENT_PATHS["todos_file"], "w") as f:
            json.dump(session_state.get("todos", []), f, indent=2)

        # Save session metadata (global step counter)
        metadata = {
            "global_step_counter": session_state.get("global_step_counter", 0),
            "selected_model": session_state.get("selected_model", "Unknown"),
            "raw_data_source": session_state.get("selected_raw_data_source", ""),
            "representation_type": session_state.get("representation_type", session_state.get("selected_model", "1D_CNN")),
            "finetune_calls_this_session": session_state.get("finetune_calls_this_session", 0),
        }
        with open(PERSISTENT_PATHS["session_metadata"], "w") as f:
            json.dump(metadata, f, indent=2)

        # Handle memory saving for both dict (headless) and object (streamlit) states
        if isinstance(session_state, dict):
            if "memory" in session_state:
                session_state["memory"].save()
            
            # --- FIX: Save Strategic Graph Memory ---
            if "strategic_graph_memory" in session_state:
                try:
                    session_state["strategic_graph_memory"].save()
                except Exception as e:
                    log(f"[State Manager] WARNING: Failed to save Strategic Graph Memory: {e}")
            # ----------------------------------------
        else:
            if hasattr(session_state, "memory"):
                session_state.memory.save()
            
            # --- FIX: Save Strategic Graph Memory ---
            if hasattr(session_state, "strategic_graph_memory"):
                try:
                    session_state.strategic_graph_memory.save()
                except Exception as e:
                    log(f"[State Manager] WARNING: Failed to save Strategic Graph Memory: {e}")
            # ----------------------------------------

        log("[State Manager] State and results saved successfully.")
    except Exception as e:
        log(f"[State Manager] ERROR: Failed to save state: {e}")


def load_state(session_state):
    """Loads the session state from disk, if the files exist."""
    try:
        log("[State Manager] Attempting to load state...")
        _ensure_dirs_exist()

        results_path = PERSISTENT_PATHS["results_log"]
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                content = f.read()
                if content:
                    session_state.results_log = json.loads(content)

        convo_path = PERSISTENT_PATHS["conversation_history"]
        if os.path.exists(convo_path):
            with open(convo_path, "r") as f:
                content = f.read()
                if content:
                    session_state.conversation_history = json.loads(content)
        strategic_path = PERSISTENT_PATHS["strategic_log"]
        if os.path.exists(strategic_path):
            with open(strategic_path, "r") as f:
                content = f.read()
                if content:
                    session_state.strategic_log = json.loads(content)
                    log("[State Manager] Strategic log loaded successfully.")

        flags_path = PERSISTENT_PATHS["architecture_flags"]
        if os.path.exists(flags_path):
            with open(flags_path, "r") as f:
                content = f.read()
                if content:
                    session_state.architecture_flags = json.loads(content)
                    log("[State Manager] Architecture flags loaded.")

        todos_path = PERSISTENT_PATHS["todos_file"]
        if os.path.exists(todos_path):
            with open(todos_path, "r") as f:
                content = f.read()
                if content:
                    session_state.todos = json.loads(content)
                    session_state.todos = json.loads(content)
                    log("[State Manager] To-Do list loaded.")

        metadata_path = PERSISTENT_PATHS["session_metadata"]
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                content = f.read()
                if content:
                    metadata = json.loads(content)
                    session_state.global_step_counter = metadata.get("global_step_counter", 0)
                    log(f"[State Manager] Global step counter loaded: {session_state.global_step_counter}")
                    saved_model = metadata.get("selected_model")
                    if saved_model and saved_model != "Unknown":
                        session_state.selected_model = saved_model
                        log(f"[State Manager] Restored selected model from disk: {saved_model}")
                    session_state._saved_raw_data_source = metadata.get("raw_data_source", "")
                    saved_rep_type = metadata.get("representation_type")
                    if saved_rep_type:
                        session_state.representation_type = saved_rep_type
                        log(f"[State Manager] Restored representation_type from disk: {saved_rep_type}")
                    
        # --- RESTORE FINETUNE COUNTER WITH SELF-HEALING FALLBACK ---
        finetune_calls = metadata.get("finetune_calls_this_session", 0)
        if finetune_calls == 0 and os.path.exists(PERSISTENT_PATHS["results_log"]):
            try:
                with open(PERSISTENT_PATHS["results_log"], "r") as results_f:
                    results_content = results_f.read()
                    if results_content:
                        results = json.loads(results_content)
                        # Count quanti trial di tipo finetune sono stati completati
                        finetune_calls = sum(
                            1 for r in results 
                            if isinstance(r, dict) and r.get("source") == "finetune_trial"
                        )
                        if finetune_calls > 0:
                            log(f"[State Manager] Self-healing fallback: recovered {finetune_calls} finetune trials from results_log.")
            except Exception as e:
                log(f"[State Manager Warning] Fallback log scanning failed: {e}")
        
        if isinstance(session_state, dict):
            session_state["finetune_calls_this_session"] = finetune_calls
        else:
            try:
                session_state.finetune_calls_this_session = finetune_calls
            except Exception:
                session_state["finetune_calls_this_session"] = finetune_calls
        log(f"[State Manager] Restored finetune_calls_this_session: {finetune_calls}")
    except Exception as e:
        log(f"[State Manager] WARNING: Could not load state: {e}")


def log_step_performance(session_state, step_data: dict):
    """
    Appends a new row to the detailed step-by-step performance CSV log.
    The log file is unique for each session.
    """
    _ensure_dirs_exist()

    log_filename = session_state.get("performance_log_file")
    if not log_filename:
        log("[State Manager] ERROR: Session-specific log filename not found.")
        return

    performance_log_dir = os.path.join(PERSISTENT_PATHS["logs_dir"], "performance_sessions")
    os.makedirs(performance_log_dir, exist_ok=True)
    performance_log_path = os.path.join(performance_log_dir, log_filename)

    headers = [
        "timestamp",
        "step",
        "architecture",
        "manifest_name",
        "memory_mode",
        "tool_used",
        "tool_status",
        "trial_accuracy",
        "best_accuracy_so_far",
        "reward",
        "num_conv_layers",
        "filters",
        "kernel_size",
        "dense_units",
        "dropout_rate",
        "learning_rate",
        "batch_size",
    ]

    file_exists = os.path.exists(performance_log_path)

    try:
        with open(performance_log_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
                log(f"Created new session performance log: {performance_log_path}")
            writer.writerow(step_data)
    except Exception as e:
        log(f"[State Manager] ERROR: Could not write to performance log: {e}")


def archive_conversation_log(session_state, conversation_history: list):
    """
    Saves the current conversation to a unique, timestamped file
    in the logs/archive directory instead of deleting it.
    """
    if not conversation_history or len(conversation_history) < 2:  # Non salvare log vuoti o di un solo passo
        log("[State Manager] Conversation history is empty, nothing to archive.")
        return

    try:
        # Crea la cartella degli archivi se non esiste
        archive_dir = os.path.join(PERSISTENT_PATHS["logs_dir"], "conversation_archives")
        os.makedirs(archive_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Tenta di ottenere informazioni di contesto per un nome file migliore
        model_name = "unknown_model"
        step = "unknown_step"
        # Use the mock st.session_state directly
        if "selected_model" in session_state:
            model_name = session_state.get("selected_model", model_name).replace(".py", "")
        if "global_step_counter" in session_state:
            step = session_state.get("global_step_counter", step)

        filename = f"convo_{model_name}_step{step}_{timestamp}.json"
        archive_path = os.path.join(archive_dir, filename)

        with open(archive_path, "w") as f:
            json.dump(conversation_history, f, indent=2)

        log(f"[State Manager] Conversation log (len {len(conversation_history)}) archived to {filename}")

    except Exception as e:
        log(f"[State Manager] ERROR: Failed to archive conversation log: {e}")


def reset_state(session_state, clear_disk: bool = False):
    """
    Resets the session state. A "soft" reset clears the UI elements,
    while a "hard" reset (clear_disk=True) deletes all data on disk.
    """
    log(f"[State Manager] Resetting session state. Clear disk: {clear_disk}")

    keys_to_reset = [
        "conversation_history",
        "todos",
    ]

    for key in keys_to_reset:
        if key in session_state:
            del session_state[key]
            
    if isinstance(session_state, dict):
        session_state["finetune_calls_this_session"] = 0
    else:
        try:
            session_state.finetune_calls_this_session = 0
        except Exception:
            session_state["finetune_calls_this_session"] = 0

    log("Session state has been reset.")

def get_todos(session_state):
    """Retrieves the current To-Do list from the session state."""
    if isinstance(session_state, dict):
        return session_state.get("todos", [])
    return getattr(session_state, "todos", [])

def update_todos(session_state, new_todos):
    """Updates the To-Do list in the session state and saves it."""
    if isinstance(session_state, dict):
        session_state["todos"] = new_todos
    else:
        session_state.todos = new_todos
    save_state(session_state)
