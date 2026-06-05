# rl_utils.py
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from tools import AVAILABLE_TOOLS, MODEL_BUILDERS

# --- INIZIO MODIFICA ---
ALL_AVAILABLE_TOOLS = list(AVAILABLE_TOOLS.keys())
# 1. Aggiungiamo un placeholder per rappresentare "qualsiasi architettura custom"
ALL_MODEL_ARCHITECTURES = list(MODEL_BUILDERS.keys()) + ["CUSTOM_ARCHITECTURE"]

# 2. Aggiorniamo la dimensione dello stato di conseguenza
# Esempio: 7 modelli standard + 1 custom + 3 metriche di performance
STATE_SIZE = len(ALL_AVAILABLE_TOOLS) + len(ALL_MODEL_ARCHITECTURES) + 3 + 2 + 1  # +1 for dataset hash


def encode_state(session_state):
    """
    Codifica lo stato attuale dell'applicazione in un vettore numerico.
    Gestisce correttamente le architetture standard e custom.
    """
    # 1. Architettura Attuale (rappresentata con One-Hot Encoding)
    arch_vector = np.zeros(len(ALL_MODEL_ARCHITECTURES))
    current_arch = session_state.get("selected_model")

    try:
        # Prova a trovare il modello nella lista di quelli standard
        current_arch_index = ALL_MODEL_ARCHITECTURES.index(current_arch)
        arch_vector[current_arch_index] = 1.0
    except ValueError:
        # Se non lo trova, è un'architettura custom. Attiviamo il bit 'CUSTOM_ARCHITECTURE'.
        custom_index = ALL_MODEL_ARCHITECTURES.index("CUSTOM_ARCHITECTURE")
        arch_vector[custom_index] = 1.0

    # --- FINE MODIFICA ---

    # 2. Performance Recente (logica invariata, ma ora lavora con uno stato di base corretto)
    results_df = pd.DataFrame(session_state.get("results_log", []))
    global_best_accuracy = results_df["mean_accuracy"].max() if not results_df.empty else 0.0

    arch_df = (
        results_df[results_df["architecture"] == current_arch]
        if not results_df.empty and current_arch
        else pd.DataFrame()
    )

    if not arch_df.empty:
        # NaN-guard: idxmax() returns NaN when the column is entirely NaN (e.g. divergent
        # trials wrote NaN mean_accuracy), and .loc[NaN] then raises KeyError: nan inside
        # encode_state — which is called before EVERY tool, deadlocking the agent.
        # Drop NaN before idxmax and fall back to safe defaults if nothing valid remains.
        valid_acc = arch_df["mean_accuracy"].dropna()
        if not valid_acc.empty:
            best_arch_accuracy = float(valid_acc.max())
            last_5 = valid_acc.tail(5)
            trend = np.polyfit(range(len(last_5)), last_5, 1)[0] if len(last_5) > 1 else 0.0
            trials_count = len(arch_df) / 50.0
            best_trial_for_arch = arch_df.loc[valid_acc.idxmax()]
            stability = best_trial_for_arch.get("std_accuracy", 1.0) or 1.0
        else:
            best_arch_accuracy = 0.0
            trend = 0.0
            trials_count = len(arch_df) / 50.0
            stability = 1.0
    else:
        best_arch_accuracy = 0.0
        trend = 0.0
        trials_count = 0.0
        stability = 1.0

    perf_vector = np.array(
        [
            round(best_arch_accuracy, 4),
            round(trend, 4),
            round(trials_count, 3),
            round(stability, 4),  # <-- Aggiunto
            round(global_best_accuracy, 4),
        ]
    )
    # 4. Aggiungi l'ultima azione eseguita (One-Hot Encoding)
    last_tool_vector = np.zeros(len(ALL_AVAILABLE_TOOLS))
    last_tool = session_state.get("last_tool_used", None)
    if last_tool and last_tool in ALL_AVAILABLE_TOOLS:
        last_tool_index = ALL_AVAILABLE_TOOLS.index(last_tool)
        last_tool_vector[last_tool_index] = 1.0

    # 5. Dataset context — differentiates experiments across datasets
    # A normalized hash ensures SERS and CIFAR-10 produce different anchor nodes in the graph,
    # preventing Q-value contamination across datasets.
    dataset_id = session_state.get("raw_data_source", "unknown")
    dataset_hash = float(hash(dataset_id) % 10000) / 10000.0
    dataset_vector = np.array([dataset_hash])

    # 6. Unisci TUTTI i vettori
    state_vector = np.concatenate([arch_vector, perf_vector, last_tool_vector, dataset_vector])

    return np.reshape(state_vector, [1, state_vector.shape[0]])  # Usa la forma dinamica


def calculate_reward(tool_name: str, tool_result: dict, accuracy_before: float, accuracy_after: float):
    """
    Calculates the immediate reward 'r' using a Potential-Based Reward Shaping approach.
    It rewards progress towards a user-defined optimization target.
    """
    if tool_result.get("status") == "error":
        return -1.0  # Max penalty for any error

    # Handle non-experimental tools
    if tool_name == "recall_relevant_memories":
        memories = tool_result.get("memories")
        return 0.1 if memories and "No relevant memories" not in str(memories) else -0.05

    # For other analytical tools that don't change accuracy
    if accuracy_after == accuracy_before:
        # Small penalty for using a step without making progress
        return -0.05

    # --- Potential-Based Reward Shaping Logic ---
    target_accuracy = st.session_state.get("optimization_target", 0.99)

    # Potential is the negative distance to the goal
    potential_before = -abs(target_accuracy - accuracy_before)
    potential_after = -abs(target_accuracy - accuracy_after)

    # The reward is the change in potential
    reward = potential_after - potential_before

    # Add a bonus if the target is reached or surpassed
    if accuracy_after >= target_accuracy:
        reward += 1.0

    return reward


from system_utils import get_gpu_memory_info


def calculate_ranker_reward(agent_thought_text: str, retrieved_memories: List[Dict]) -> List[float]:
    """
    Calculates a reward for each retrieved memory based on whether it was "cited"
    in the agent's subsequent thoughts.
    """
    rewards = []
    # A simple way to avoid matching very short, common words
    MIN_SUBSTRING_LEN = 15

    for memory in retrieved_memories:
        finding_text = memory.get("finding", "")

        # Simple check: was a significant part of the memory text used in the agent's thoughts?
        # This is a proxy for "explicit citation".
        if finding_text in agent_thought_text:
            # High reward if the whole finding is present
            rewards.append(1.0)
        else:
            # Check for smaller, significant substrings
            cited = False
            # Break the finding into smaller, overlapping chunks to check for partial citation
            chunks = [
                finding_text[i : i + MIN_SUBSTRING_LEN] for i in range(0, len(finding_text) - MIN_SUBSTRING_LEN, 5)
            ]
            for chunk in chunks:
                if chunk in agent_thought_text:
                    cited = True
                    break

            if cited:
                rewards.append(1.0)  # High reward for being cited
            else:
                rewards.append(-0.1)  # Small penalty for being ignored (noise)

    return rewards


from typing import Any  # Ensure Any is available if not already

def update_graph_memory(thread_safe_state: Dict[str, Any], tool_name: str, tool_result: dict, state_before, state_after, tool_args: Dict[str, Any] = None):
    """
    Updates the Strategic Graph Memory with the experience from the last tool execution.
    Calculates the reward based on the state transition and updates the Q-values.
    """
    from ui_logger import log

    if "strategic_graph_memory" not in thread_safe_state:
        log("[Graph Memory] Warning: 'strategic_graph_memory' not in state. Skipping update.")
        return

    memory_system = thread_safe_state["strategic_graph_memory"]
    
    # 1. Calculate Reward
    # Note: state_before and state_after are encoded vectors (np.arrays).
    # Based on encode_state:
    # Index [0:N_ARCH] is encoding
    # Index [N_ARCH] is best_arch_accuracy (Perf[0])
    
    # Calculate index dynamically
    n_archs = len(ALL_MODEL_ARCHITECTURES)
    
    # Perf vector starts after Arch vector
    # best_arch_accuracy is the first element of the Perf vector
    idx_acc = n_archs
    
    try:
        accuracy_before = float(state_before[0, idx_acc])
        accuracy_after = float(state_after[0, idx_acc])
    except (IndexError, TypeError):
        accuracy_before = 0.0
        accuracy_after = 0.0
    
    reward = calculate_reward(tool_name, tool_result, accuracy_before, accuracy_after)
    
    # 2. Extract Context
    current_arch = thread_safe_state.get("selected_model", "unknown")
    # If the tool was called with a custom_architecture_file, use it as the arch label
    # so the graph records experience under the custom arch, not the base representation model.
    if tool_args and tool_args.get("custom_architecture_file"):
        current_arch = tool_args["custom_architecture_file"]

    # For 'next_architecture', usually it's the same unless we switched.
    next_arch = current_arch
    if isinstance(tool_result, dict) and "updated_selected_model" in tool_result:
        next_arch = tool_result["updated_selected_model"]
        
    # 3. Identify Finding
    finding_id = None
    if isinstance(tool_result, dict) and "finding_id" in tool_result:
        finding_id = tool_result["finding_id"]

    # 4. Construct Rich Action Name
    # We filter out large/internal args to keep the graph readable but informative
    action_repr = tool_name
    full_arguments = {}
    
    if tool_args:
        # Filter logic: Exclude internal keys AND callable objects (functions cannot be serialized)
        ignored_keys = ["thread_safe_state", "agent", "code", "file_content", "content", "system_prompt", "execute_tool_func"]
        filtered_args = {
            k: v for k, v in tool_args.items() 
            if k not in ignored_keys and not callable(v)
        }
        
        # We save this dictionary as the FULL arguments for replayability
        full_arguments = filtered_args.copy()

        # --- DATASET METADATA INJECTION ---
        # Look up what data we are actually working on here from thread_safe_state
        selected_source = thread_safe_state.get("selected_raw_data_source", "unknown")
        
        # Try to find manifest name from full_arguments (which has filtered kwargs of the tool)
        # For instance, run_training_trial receives 'manifest_name'
        current_manifest = full_arguments.get("manifest_name")
        
        if not current_manifest:
             # Fallback 1: Extract from tool result if available
             if isinstance(tool_result, dict) and "manifest_name" in tool_result:
                 current_manifest = tool_result["manifest_name"]
             # Fallback 2: Extract from latest results log
             elif "results_log" in thread_safe_state and thread_safe_state["results_log"]:
                 current_manifest = thread_safe_state["results_log"][-1].get("manifest_name")
             else:
                 current_manifest = "unknown_manifest"

        full_arguments["_data_source"] = selected_source
        full_arguments["_manifest_name"] = current_manifest
        
        # Now try to grab the exact parameters for this manifest to embed them
        try:
            from tools import _load_dataset_metadata
            md = _load_dataset_metadata(thread_safe_state)
            if current_manifest and current_manifest in md:
                params_dict = {k: v for k, v in md[current_manifest].get("params", {}).items() if k != "raw_data_path"}
                full_arguments["_dataset_params"] = params_dict
        except Exception as e:
            log(f"[Graph Memory WARNING] Could not fetch dataset params for manifest '{current_manifest}': {e}")
        # ----------------------------------

        # Summarize known complex args like 'params_json' if strings
        for k, v in filtered_args.items():
             if k == "params_json" and isinstance(v, str):
                 try:
                     # try to minify json string
                     filtered_args[k] = json.loads(v)
                 except:
                     pass

        if filtered_args:
             # Create a compact string representation
             # e.g. "run_training (epochs=10, lr=0.01)"
             args_str = ", ".join([f"{k}={v}" for k, v in filtered_args.items()])
             # Cap length to avoid massive edge keys
             if len(args_str) > 50:
                 args_str = args_str[:47] + "..."
             action_repr = f"{tool_name}({args_str})"
    
    # 5. Populate Multi-Objective Impact Vector
    # [0]: Delta Acc, [1]: Overfit, [2]: Speed, [3]: Latency, [4]: Memory
    impact_vector = [0.0] * 5
    
    # [0] Accuracy Progress (Core Utility)
    impact_vector[0] = accuracy_after - accuracy_before
    
    if tool_result.get("status") == "completed":
        # Extract metrics from training_worker report
        impact_vector[1] = tool_result.get("mean_overfitting_score", 0.0)
        impact_vector[2] = tool_result.get("mean_learning_speed", 0.0)
        impact_vector[3] = tool_result.get("inference_time_ms", 0.0)
        impact_vector[4] = tool_result.get("model_size_mb", 0.0)
    
    # Fallback to Shaped Reward for Accuracy index if accuracy_after is not provided or same
    # This keeps analytical tools (which return reward= -0.05) visible in the graph.
    if impact_vector[0] == 0:
        impact_vector[0] = reward
    
    try:
        memory_system.add_organic_experience(
            state_before=state_before,
            state_after=state_after,
            action_name=action_repr,
            impact_vector=impact_vector,
            current_architecture=current_arch,
            finding_id=finding_id,
            full_arguments=full_arguments
        )
        # log(f"[MOSAN] Experience recorded. Impact: {impact_vector}")
    except Exception as e:
        log(f"[Graph Memory ERROR] Failed to update experience: {e}")

