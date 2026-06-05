# cluster/main.py
import os
import sys

# --- FRAMEWORK STABILITY FIX (Entry Point) ---
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ---------------------------------------------

import asyncio
import json
import os
import re
import subprocess
import sys
import os
import argparse
import logging
import traceback

# --- CRITICAL FIX: Explicit GPU Initialization ---
try:
    # Try importing from root directly (if running from root)
    from system_utils import setup_gpu_memory
    setup_gpu_memory()
except ImportError:
    # If starting from root without cluster package in path
    sys.path.append(os.getcwd())
    try:
        from system_utils import setup_gpu_memory
        setup_gpu_memory()
    except ImportError as e:
        print(f"Warning: Could not import system_utils for GPU setup: {e}")
# -------------------------------------------------
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import tensorflow as tf

# --- Single Ledger Architecture (Phase 1 + Phase 2 + Phase 3) ---
from ledger_schema import create_ledger, get_latest_ledger, read_ledger, SECTION_MARKERS
from analyst import run_analyst
from theorist import run_theorist
from decider_ledger import run_decider
from summarizer_ledger import run_multi_agent_summarizer
from user_directive import read_user_directive

# --- HITL Import ---
from cluster.core.dynamic_loader import DynamicToolLoader
from cluster.core.tool_error_handler import handle_tool_execution_error
from cluster.core.state_proxy import SharedStateManager
# -------------------

# Disable JIT/XLA globally to prevent "Graph execution error"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
tf.config.optimizer.set_jit(False)

# Debug / Stability: Log Env and Enable Memory Growth
try:
    print(f"[Init] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
    gpus = tf.config.list_physical_devices('GPU')
    print(f"[Init] Physical GPUs found: {len(gpus)}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("[Init] GPU Memory Growth Enabled.")
except Exception as e:
    print(f"[Init Warning] Failed to setup GPU memory growth: {e}")

# --- Mock Streamlit (Conditional) ---
class MockSessionState(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'MockSessionState' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

class MockStreamlit:
    def __init__(self):
        self.session_state = MockSessionState()
    def set_page_config(self, *args, **kwargs): pass
    def sidebar(self): return self
    def title(self, *args, **kwargs): pass
    def markdown(self, *args, **kwargs): pass
    def divider(self, *args, **kwargs): pass
    def header(self, *args, **kwargs): pass
    def radio(self, *args, **kwargs): pass
    def text_input(self, *args, **kwargs): pass
    def button(self, *args, **kwargs): return False
    def warning(self, *args, **kwargs): print(f"[UI WARNING] {args[0]}")
    def error(self, *args, **kwargs): print(f"[UI ERROR] {args[0]}")
    def success(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def toast(self, *args, **kwargs): pass
    def container(self, *args, **kwargs): return self
    def chat_message(self, *args, **kwargs): return self
    def expander(self, *args, **kwargs): return self
    def number_input(self, *args, **kwargs): return 0
    def selectbox(self, *args, **kwargs): return None
    def multiselect(self, *args, **kwargs): return []
    def metric(self, *args, **kwargs): pass
    def json(self, *args, **kwargs): pass
    def dataframe(self, *args, **kwargs): pass
    def image(self, *args, **kwargs): pass
    def file_uploader(self, *args, **kwargs): return None
    def checkbox(self, *args, **kwargs): return False
    def rerun(self, *args, **kwargs): pass
    def chat_input(self, *args, **kwargs): return None
    def spinner(self, *args, **kwargs): return self
    def __enter__(self): return self
    def __exit__(self, *args, **kwargs): pass

# Determine if we should mock Streamlit
use_mock = True
try:
    import streamlit as _real_st
    # Check if we are running within a Streamlit Runtime
    # (Works in Streamlit >= 1.14)
    from streamlit.runtime import exists as _st_exists
    if _st_exists():
        use_mock = False
        st = _real_st
        print("[Init] Streamlit runtime detected. Using Real Streamlit.")
    else:
        # Detected library but no runtime (e.g. 'python main.py')
        print("[Init] Streamlit library found but no runtime. Defaulting to Mock.")
except Exception as e:
    print(f"[Init] Streamlit detection failed: {e}. Defaulting to Mock.")

if use_mock:
    st = MockStreamlit()
    sys.modules["streamlit"] = st
    print("[Init] Using MOCK Streamlit (Headless Mode).")

import hydra
import numpy as np
import pandas as pd
import tensorflow as tf
from ddgs import DDGS
from omegaconf import DictConfig

from agent import ConversationalAgent, DeciderOrchestrator
from graph_memory import StrategicGraphMemory
from memory_utils import VectorMemory
from rl_utils import calculate_ranker_reward, calculate_reward, encode_state
from state_manager import (
    PERSISTENT_PATHS,
    archive_conversation_log,
    load_state,
    log_step_performance,
    save_state,
    get_todos,
)
from tools import AVAILABLE_TOOLS, MODEL_BUILDERS
from cluster.prompts import LEDGER_DECIDER_SYSTEM_PROMPT as SYSTEM_PROMPT
from ui_logger import init_log, log
from exceptions import AgentCriticalError

# Change working directory to the script's directory
# os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- Utility Functions ---
def convert_numpy_types(obj):
    """
    Recursively convert NumPy/Pandas types to native Python types for JSON serialization.
    
    Handles:
    - np.bool_ → bool
    - np.integer (int64, int32, etc.) → int
    - np.floating (float64, float32, etc.) → float
    - np.ndarray → list
    - pd.DataFrame → dict (via to_dict())
    - pd.Series → list (via to_list())
    - datetime → ISO format string
    """
    import datetime
    
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    # CRITICAL FIX: Handle np.bool_ BEFORE other numpy types
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Pandas support (optional, for future-proofing)
    elif hasattr(obj, 'to_dict'):  # DataFrame-like
        return convert_numpy_types(obj.to_dict())
    elif hasattr(obj, 'to_list'):  # Series-like
        return convert_numpy_types(obj.to_list())
    # Datetime support
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    else:
        return obj


def reinitialize_agent(cfg: DictConfig):
    provider = cfg.api.provider
    if provider == "LM Studio (Local)":
        base_url, api_key, model_name = "http://host.docker.internal:1234/v1", "not-needed", "local-model"
    elif provider == "vllm":
        base_url = cfg.api.get("local_llm_url", "http://localhost:8000/v1")
        api_key = "EMPTY"
        # Dynamically fetch the model name from vLLM with retries
        model_name = "local-model"
        for attempt in range(5):
            try:
                import requests
                response = requests.get(f"{base_url}/models", timeout=5)
                if response.status_code == 200:
                    model_name = response.json()["data"][0]["id"]
                    log(f"Detected vLLM model: {model_name}")
                    break
                else:
                    log(f"Warning: vLLM returned status {response.status_code}. Retrying ({attempt+1}/5)...")
            except Exception as e:
                log(f"Warning: Error connecting to vLLM: {e}. Retrying ({attempt+1}/5)...")
            time.sleep(5)
        if model_name == "local-model":
             log("ERROR: Could not fetch model name from vLLM after retries. Defaulting to 'local-model' (Expect 404s).")
    else:
        base_url = "https://openrouter.ai/api/v1"
        api_key = cfg.api.openrouter_api_key
        model_name = cfg.api.openrouter_model_name
    st.session_state.agent = ConversationalAgent(base_url=base_url, api_key=api_key, model_name=model_name)
    
    # --- Innovation Team Model Config ---
    # Allow overriding the coding model via config (e.g. for specialized coding LLMs)
    coding_model = cfg.api.get("coding_llm_model", None)
    if coding_model:
        st.session_state.coding_model_name = coding_model
        
        # HYBRID PROVIDER SUPPORT:
        # If the main provider is NOT OpenRouter, but we want to use an OpenRouter model for coding,
        # we must instantiate a dedicated agent with the correct Basic URL/Key.
        # We assume if 'coding_llm_model' is set, it might require OpenRouter.
        # To be safe, we check if the model name looks like an external ID or if the user explicitly wants OpenRouter.
        
        # Simplify: If coding_model is set, we CREATE A DEDICATED AGENT pointing to OpenRouter
        # This is robust even if the main agent is also OpenRouter (just a separate client).
        
        log(f"[System] Initializing dedicated Coding Agent for model: {coding_model}")
        try:
            if provider == "vllm":
                coding_agent = ConversationalAgent(base_url=base_url, api_key=api_key, model_name=model_name)
                st.session_state.coding_agent = coding_agent
                log(f"[System] Coding Agent initialized successfully (Provider: vllm, model: {model_name}).")
            else:
                or_base_url = "https://openrouter.ai/api/v1"
                or_api_key = cfg.api.get("openrouter_api_key")
                if not or_api_key or "sk-" not in or_api_key:
                    log("[System] WARNING: coding_llm_model set but openrouter_api_key missing/invalid. Fallback to main agent.")
                    st.session_state.coding_agent = None
                else:
                    coding_agent = ConversationalAgent(base_url=or_base_url, api_key=or_api_key, model_name=coding_model)
                    st.session_state.coding_agent = coding_agent
                    log(f"[System] Coding Agent initialized successfully (Provider: OpenRouter).")
        except Exception as e:
            log(f"[System] Failed to initialize Coding Agent: {e}")
            st.session_state.coding_agent = None
    else:
        if "coding_model_name" not in st.session_state:
             st.session_state.coding_model_name = None
             st.session_state.coding_agent = None
        log("[System] Innovation Team using default agent model.")
    # ------------------------------------

    # --- Theorist / Decider split-model config ---
    # theorist_model: thinking model for deep hypothesis reasoning (e.g. qwen/qwen3.6-plus)
    # decider_model: fast non-thinking model for reliable tool calls (e.g. qwen/qwen3-235b-a22b)
    # vllm: single local model serves all roles — no separate agents needed.
    # OpenRouter/LM Studio: create dedicated agents per role if model differs from main.
    for role, cfg_key, state_key in [
        ("Theorist", "theorist_model", "theorist_agent"),
        ("Decider",  "decider_model",  "decider_agent"),
    ]:
        role_model = cfg.api.get(cfg_key, None)
        if provider == "vllm":
            # All roles share the single vllm model — no dedicated agent needed.
            st.session_state[state_key] = None
            if role_model:
                log(f"[System] {role} using vllm main agent ({model_name}).")
        elif role_model and role_model != model_name:
            try:
                or_base_url = "https://openrouter.ai/api/v1"
                or_api_key = cfg.api.get("openrouter_api_key", api_key)
                role_agent = ConversationalAgent(base_url=or_base_url, api_key=or_api_key, model_name=role_model)
                st.session_state[state_key] = role_agent
                log(f"[System] {role} Agent initialized: {role_model} (via OpenRouter)")
            except Exception as e:
                log(f"[System] Failed to initialize {role} Agent ({role_model}): {e}. Falling back to main agent.")
                st.session_state[state_key] = None
        else:
            st.session_state[state_key] = None
            if role_model:
                log(f"[System] {role} using main agent ({model_name}).")
    # --------------------------------------------

    log(f"Agent connected to {provider}!")

def update_best_result_from_log():
    log("[APP] Recalculating best result from log.")
    results_log = st.session_state.get("results_log", [])
    if not results_log:
        st.session_state.best_accuracy = 0.0
        st.session_state.best_params = {}
        return
    best_trial = max(results_log, key=lambda x: x.get("mean_accuracy") or 0.0)
    st.session_state.best_accuracy = best_trial.get("mean_accuracy") or 0.0
    best_params = best_trial.get("params", {})
    st.session_state.best_params = convert_numpy_types(best_params)
    log(f"[APP] Best result recalculated. Accuracy: {st.session_state.best_accuracy:.4f}")

def _run_duckduckgo_search(queries: list[str], max_results_per_query: int = 5) -> list[dict]:
    log(f"[Search Tool] Running DDGS for queries: {queries}")
    all_results = []
    with DDGS(timeout=20) as ddgs:
        for query in queries:
            try:
                results = ddgs.text(query, max_results=max_results_per_query)
                if results:
                    for r in results:
                        all_results.append({"source_title": r.get("title", "No Title"), "url": r.get("href", ""), "snippet": r.get("body", "")})
            except Exception as e:
                log(f"[Search Tool ERROR] Failed to search for query '{query}': {e}")
                continue
    return all_results

def execute_tool(tool_name: str, tool_args: Dict[str, Any], thread_safe_state: Dict[str, Any], gpu_id: Optional[int] = None, execute_tool_func: Optional[Callable] = None, depth: int = 0) -> Dict[str, Any]:
    MAX_RECURSION_DEPTH = 5
    if depth > MAX_RECURSION_DEPTH:
        log(f"[Recursion Guard] 🛑 Exception: Depth {depth} exceeded max {MAX_RECURSION_DEPTH}. Stopping.")
        return {"status": "error", "message": f"Recursion limit ({MAX_RECURSION_DEPTH}) exceeded."}

    if tool_name in AVAILABLE_TOOLS:
        log(f"--- EXECUTING TOOL: {tool_name} (Depth: {depth}, GPU: {gpu_id if gpu_id is not None else 'N/A'}) ---")
        log(f"Received arguments: {tool_args}")
        tool_args["thread_safe_state"] = thread_safe_state
        selected_model = thread_safe_state.get("selected_model")

        if tool_name in ["run_training_trial", "run_strategic_optuna_sweep", "run_supernet_search"]:
            # --- FIX: ROBUST SUBSTITUTION FOR CUSTOM ARCHITECTURES ---
            # If selected_model is a python file (e.g. 'my_model.py'), we must treat it as a custom architecture.
            if selected_model and selected_model.endswith(".py"):
                tool_args["custom_architecture_file"] = selected_model
                # We interpret this as "Use this code".
                # But we MUST provide a representation_type. 
                # If not provided in args, we default to '1D_CNN' or the last used one.
                if "representation_type" not in tool_args:
                     # Attempt to infer from state or default
                     tool_args["representation_type"] = "1D_CNN" # Safe default for audio/SERS
                     log(f"[Tool Wrapper] Auto-mapped '{selected_model}' to custom_architecture_file with default rep '1D_CNN'")
            else:
                # Standard behavior (e.g. '1D_CNN')
                tool_args["experiment_name"] = tool_args.get("custom_architecture_file") or selected_model
            
            if gpu_id is not None:
                tool_args["gpu_id"] = gpu_id
            # ---------------------------------------------------------
        if tool_name in ["memorize_finding", "get_architecture_summary"]:
            tool_args["experiment_name"] = selected_model
        if tool_name in ["propose_intelligent_architecture", "run_supernet_search"]:
            tool_args["agent"] = thread_safe_state.get("agent")
            tool_args["depth"] = depth
        if tool_name == "run_web_research":
            tool_args["google_search_tool"] = _run_duckduckgo_search
        
        # Inject execute_tool_func for tools that need recursion
        TOOLS_REQUIRING_EXECUTE_FUNC = [
            "run_architecture_comparison", 
            "run_strategic_graph_evaluation", 
            "generate_scientific_report", 
            "generate_finetuning_dataset"
        ]
        if tool_name in TOOLS_REQUIRING_EXECUTE_FUNC and execute_tool_func:
            # Create a wrapper that increments depth for the next level
            def depth_aware_wrapper(n, a, s, g=None):
                # We call the original executor (recursive_execute_tool) but tell it to pass the new depth
                # Wait, recursive_execute_tool calls execute_tool. 
                # Use the 'depth' arg we added to execute_tool.
                return execute_tool(n, a, s, g, execute_tool_func=execute_tool_func, depth=depth + 1)
            
            tool_args["execute_tool_func"] = depth_aware_wrapper

        # --- MEMORY UPDATE HOOK ---
        # Capture state BEFORE tool execution
        from rl_utils import encode_state, update_graph_memory
        state_before = encode_state(thread_safe_state)
        # --------------------------

        try:
            result = AVAILABLE_TOOLS[tool_name](**tool_args)
        except Exception as e:
            # --- SMART ERROR HANDLING ---
            log(f"[Tool Error] Caught exception for {tool_name}: {e}")
            result = handle_tool_execution_error(tool_name, e, AVAILABLE_TOOLS[tool_name])
            # ----------------------------

        serializable_result = convert_numpy_types(result)
        
        # --- LOGGING TOOL RESULT (USER REQUEST) ---
        try:
            log(f"--- TOOL RESULT: {tool_name} ---\n{json.dumps(serializable_result, indent=2)}")
        except Exception as e:
            log(f"--- TOOL RESULT: {tool_name} (JSON Serialization Failed) ---\n{str(serializable_result)}")
        # ------------------------------------------

        # --- MEMORY UPDATE HOOK (POST) ---
        try:
            # Capture state AFTER tool execution
            state_after = encode_state(thread_safe_state)
            
            # Update Graph Memory (Organic Growth + Q-Learning)
            update_graph_memory(
                thread_safe_state, 
                tool_name, 
                serializable_result, 
                state_before, 
                state_after, 
                tool_args
            )
        except Exception as e:
            log(f"[Main Hook] Failed to update graph memory: {e}")
        # ---------------------------------
        
        # --- MISSION 2: STATE INTERCEPTION & PIVOT HANDLING ---
        if isinstance(serializable_result, dict):
            if "updated_selected_model" in serializable_result:
                new_model = serializable_result["updated_selected_model"]
                log(f"[System] SWITCHING ARCHITECTURE: {st.session_state.selected_model} -> {new_model}")
                st.session_state.selected_model = new_model
            
            if "updated_architecture_flags" in serializable_result:
                flags = serializable_result["updated_architecture_flags"]
                st.session_state.architecture_flags.update(flags)
                log(f"[System] Updated architecture flags: {flags}")

            # Capture Innovation Team result so the next Theorist cycle can prioritize it.
            if tool_name == "propose_intelligent_architecture" and serializable_result.get("status") == "completed":
                arch_file = serializable_result.get("architecture_file")
                rep_type = serializable_result.get("representation_type") or thread_safe_state.get("selected_model", "unknown")
                manifest = serializable_result.get("manifest_name")
                if arch_file:
                    innovation_record = {
                        "status": "success",
                        "architecture_file": arch_file,
                        "representation_type": rep_type,
                        "manifest_name": manifest,
                    }
                    thread_safe_state["last_innovation_result"] = innovation_record
                    st.session_state["last_innovation_result"] = innovation_record
                    log(f"[System] Innovation result captured: {innovation_record}")
                
            # --- NEW PIVOT LOGIC ---
            if "trigger_summary_with_reason" in serializable_result:
                reason = serializable_result["trigger_summary_with_reason"]
                log(f"[System] Triggering Pivot Summary. Reason: {reason}")
                # We trigger the summary on the *new* model (which was just set above)
                trigger_strategic_summary_and_reset_memory(
                    thread_safe_state, 
                    architecture_filter=st.session_state.selected_model, 
                    special_pretext=reason
                )
            # -----------------------
        # -------------------------------------

        # --- HITL CHECK (Polling Mode Active) ---
        # The tool itself blocks until approval, so we just return the result (which contains feedback).
        # We assume result["status"] == "completed" after approval.
        # ----------------------------------------

        return serializable_result
    else:
        log(f"[APP] ERROR: Tool '{tool_name}' not found.")
        return {"status": "error", "message": f"Tool '{tool_name}' not found."}

def trigger_graph_maintenance(thread_safe_state: Dict[str, Any]):
    """
    Triggers strategic graph maintenance using the HybridGraphAdvisorCollaborator.
    This runs the A* pathfinding + Expert LLM analysis pipeline.
    Called every N steps (configurable via maintenance_interval in config.yaml).
    """
    log(f"[Graph Maintenance] Starting hybrid graph analysis at step {st.session_state.global_step_counter}...")
    
    try:
        # Import the Celery task
        from cluster.cognitive_tasks import run_strategic_graph_maintenance_task
        
        # Execute asynchronously via Celery
        run_id = f"step_{st.session_state.global_step_counter}"
        _active_dataset = st.session_state.get("selected_raw_data_source", "global") or "global"
        result = run_strategic_graph_maintenance_task.apply_async(args=[run_id, _active_dataset])
        
        log(f"[Graph Maintenance] Task dispatched to worker. Task ID: {result.id}")
        
        # Don't wait for result (async), just log dispatch
        return {"status": "dispatched", "task_id": result.id}
        
    except Exception as e:
        log(f"[Graph Maintenance] ERROR: Failed to trigger task. Error: {e}")
        return {"status": "error", "message": str(e)}

def trigger_strategic_summary_and_reset_memory(thread_safe_state: Dict[str, Any], architecture_filter: Optional[str] = None, reset_conversation: bool = True, special_pretext: Optional[str] = None):
    summary_context = f"for the specific architecture '{architecture_filter}'" if architecture_filter else "for the last block of research"
    log(f"[Multi-Agent Summary] Starting strategic summary {summary_context}...")
    
    tool_args = {"architecture_filter": architecture_filter} if architecture_filter else {}
    correlation_data = execute_tool("get_correlation_matrix", tool_args.copy(), thread_safe_state)
    best_worst_data = execute_tool("analyze_best_vs_worst_trials", tool_args.copy(), thread_safe_state)
    
    try:
        # --- PRETEXT INJECTION ---
        pretext_msg = ""
        if special_pretext:
            log(f"[Summary] Using special pretext: {special_pretext}")
            pretext_msg = f"\n**SPECIAL CONTEXT:** {special_pretext}\n(The research focus is shifting based on this context. Ensure the summary reflects this.)\n"
        # -------------------------

        summarizer_prompt = (
            f"You are the **Lead Research Summarizer**.\n"
            f"{pretext_msg}"
            f"Review the following research data {summary_context}.\n"
            f"**Data Sources:**\n"
            f"1. Correlation Matrix: {json.dumps(correlation_data)}\n"
            f"2. Best vs Worst Trials: {json.dumps(best_worst_data)}\n\n"
            f"**Task:**\n"
            f"Extract strictly objective facts and patterns. No speculation.\n"
            f"Focus on: Which hyperparameters correlate with high accuracy? What caused the failures?\n"
            f"Output a bulleted list of factual findings."
        )
        strategic_summary, _ = st.session_state.agent.run_turn(
            "You are a precise data analyst.", 
            [{"role": "user", "parts": [summarizer_prompt]}]
        )
        
        critic_prompt = (
            f"You are the **Research Critic**.\n"
            f"Review this summary:\n{strategic_summary}\n\n"
            f"**Task:**\n"
            f"Identify any potential overfitting risks, logical fallacies, or missing angles.\n"
            f"Are we ignoring any specific architecture flaws?\n"
            f"Provide a constructive critique."
        )
        corrective_lesson, _ = st.session_state.agent.run_turn(
            "You are a critical reviewer.", 
            [{"role": "user", "parts": [critic_prompt]}]
        )
        
        consolidator_prompt = (
            f"You are the **Strategic Consolidator**.\n"
            f"Combine the initial summary and the critique into a final **Strategic Lesson**.\n"
            f"**Initial Summary:** {strategic_summary}\n"
            f"**Critique:** {corrective_lesson}\n\n"
            f"**Task:**\n"
            f"Synthesize a concise, actionable lesson for the next optimization phase.\n"
            f"What should we keep? What should we change? What is the new hypothesis?"
        )
        final_summary_text, _ = st.session_state.agent.run_turn(
            "You are a strategic planner.", 
            [{"role": "user", "parts": [consolidator_prompt]}]
        )
        
        finding_to_memorize = f"Consolidated summary at step {st.session_state.global_step_counter} ({summary_context}):\n{final_summary_text}"
        execute_tool("memorize_finding", {"finding": finding_to_memorize}, thread_safe_state)

    except Exception as e:
        log(f"[Multi-Agent Summary] ERROR: Failed to generate summary. Error: {e}")
        final_summary_text = f"Summary failed due to error: {e}. Proceeding with memory reset."

    if reset_conversation:
        log("[STRATEGY] Archiving and clearing short-term memory...")
        archive_conversation_log(st.session_state, st.session_state.conversation_history)
        st.session_state.conversation_history = []
        continuation_prompt = f"**Context from Previous Multi-Agent Summary (Memory Cleared):**\n<summary>\n{final_summary_text}\n</summary>\n\nBased on this, proceed with your optimization plan."
        st.session_state.conversation_history.append({"role": "user", "parts": [continuation_prompt]})

async def run_agent_turn(thread_safe_state: Dict[str, Any], collaborator: DeciderOrchestrator):
    st.session_state.global_step_counter += 1
    log(f"--- Starting Global Step: {st.session_state.global_step_counter} ---")
    
    # --- PERIODIC MAINTENANCE CHECKS (Step-Based) ---
    reset_interval = st.session_state.get("memory_reset_interval", 20)
    graph_interval = st.session_state.get("graph_maintenance_interval", 10)
    
    # Memory reset and summarization (every N steps, e.g., 25)
    if st.session_state.global_step_counter > 1 and st.session_state.global_step_counter % reset_interval == 0:
        trigger_strategic_summary_and_reset_memory(thread_safe_state)
    
    # Graph maintenance (every N steps, e.g., 10)
    if (st.session_state.global_step_counter > 1 and 
        st.session_state.global_step_counter % graph_interval == 0 and
        st.session_state.get("strategic_advisor_enabled", False)):
        trigger_graph_maintenance(thread_safe_state)
    
    # --- STRATEGIC CONTEXT & ADVISOR (MOSAN) ---
    use_strategic_advisor = st.session_state.get("strategic_advisor_enabled", False)
    strategic_advice = None
    context_summary = None # Default
    
    try:
        # Import classes
        from context_planner import ContextCriticAgent, StrategicAdvisorTeam
        
        # 1. Context Critic: ALWAYS ANALYZE CONTEXT (Situation Awareness)
        # Even if we don't use the graph for advice, we likely want the critique.
        critic = ContextCriticAgent(st.session_state.agent)
        context_summary = await critic.summarize_context(
            thread_safe_state,
            None # User Request: Disable graph visibility for Context Critic for now (Optimization)
        )
        
        # 2. Strategic Advisor: Generate Advice (Attributes to Graph)
        if use_strategic_advisor and st.session_state.get("strategic_graph_memory") is not None:
            advisor = StrategicAdvisorTeam(
                st.session_state.agent,
                st.session_state.strategic_graph_memory,
                vector_memory=st.session_state.memory
            )
            
            # We pass the full state for encoding
            strategic_advice = await advisor.generate_advice(
                context_summary,
                thread_safe_state,
                default_lens=st.session_state.get("strategic_default_lens", "auto")
            )
            
            log(f"[Strategic Advisor] Advice:\n{strategic_advice}")

    except Exception as e:
        log(f"[Strategy Core] Error in Context/Advisor loop: {e}")
        log(f"[Strategy Core] Traceback: {traceback.format_exc()}")
        # Fail silently
        strategic_advice = None
        context_summary = None 
    # --- END STRATEGIC CONTEXT & ADVISOR ---

    # --- STANDARD FLOW (Decider Centric) ---
    # The Decider now receives inputs from Analyst, Strategist, AND Graph Planner
    
    # Simplified logic from app.py

    # Inject To-Do list into system prompt
    todos = get_todos(st.session_state)
    todos_context = ""
    if todos:
        todos_context = "**Current To-Do List Status:**\n"
        for i, todo in enumerate(todos):
            status_char = "[x]" if todo["status"] == "completed" else "[ ]"
            todos_context += f"- {status_char} {todo['description']} ({todo['status']})\n"
        todos_context += "\nBased on the above list and the user's request, decide the next action. Use the `write_todos` tool to update the status of existing tasks or add new ones.\n"

    # --- FIX METADATA CONTEXT ---
    # Recuperiamo la fonte dati dalla configurazione per riempire il contesto
    raw_source = st.session_state.cfg.data.raw_data_source
    metadata_info_str = f"Raw Data Source: {raw_source}"
    
    # DYNAMIC METADATA INJECTION: Load metadata.json if exists
    try:
        raw_data_dir = PERSISTENT_PATHS.get("raw_data_dir", "processed_data/raw_data")
        source_path = os.path.join(raw_data_dir, raw_source)
        metadata_file = os.path.join(source_path, "metadata.json")
        if os.path.exists(metadata_file):
             with open(metadata_file, "r") as f:
                 metadata_payload = json.load(f)
             metadata_info_str += f"\n\n--- DYNAMIC DATASET METADATA ---\n{json.dumps(metadata_payload, indent=2)}\n--------------------------------"
    except Exception as e:
        log(f"[Metadata] Warning: Failed to load dynamic metadata.json: {e}")
    
    # Eseguiamo ENTRAMBE le sostituzioni nel SYSTEM_PROMPT
    modified_system_prompt = SYSTEM_PROMPT.replace("{current_model}", st.session_state.selected_model)
    modified_system_prompt = modified_system_prompt.replace("{model_type}", st.session_state.selected_model)
    modified_system_prompt = modified_system_prompt.replace("{metadata_info}", metadata_info_str)
    # --- END FIX ---

    final_system_prompt = todos_context + modified_system_prompt

    # Collaborator passed as argument
    
    # Inject Strategic Insight if available
    strategic_insight = st.session_state.get("latest_strategic_insight", "")
    metadata_info = "Headless run metadata"
    if strategic_insight:
        metadata_info += f"\n\n**Strategic Graph Insights:**\n{strategic_insight}"

    # Construct dynamic initial context for the Analyst
    results_log = st.session_state.get("results_log", [])
    best_acc = st.session_state.get("best_accuracy", 0.0)
    
    current_arch = st.session_state.selected_model
    # Calculate best for specifically the current architecture
    current_arch_trials = [t for t in results_log if t.get("architecture") == current_arch]
    current_arch_best = max([(t.get("mean_accuracy") or 0.0) for t in current_arch_trials]) if current_arch_trials else 0.0
    
    context_parts = []
    context_parts.append(f"Global Best Accuracy (Any Model): {best_acc:.4f}")
    context_parts.append(f"Current Model ({current_arch}) Best Accuracy: {current_arch_best:.4f}")
    
    if results_log:
        last_trials = results_log[-5:]
        context_parts.append(f"Last {len(last_trials)} Trials (Architecture Agnostic):")
        for t in last_trials:
            # Handle potential missing keys gracefully
            m_name = t.get('manifest_name', 'N/A')
            arch = t.get('architecture', 'Unknown')
            acc = t.get('mean_accuracy', 0.0)
            context_parts.append(f"- [{arch}] {m_name}: Acc={acc:.4f}")
    else:
        context_parts.append("No trials completed yet (Fresh Start).")
        
    if strategic_insight:
        context_parts.append(f"\nStrategic Insight (Periodic):\n{strategic_insight}")
        
    if strategic_advice:
        context_parts.append(f"\n{strategic_advice}")
    elif context_summary: 
        # Fallback: If no advice (e.g. disabled graph but enabled critic), show context summary
        context_parts.append(f"\n**Context Analysis:**\nState: {context_summary.get('task_type', 'Unknown')}\nFocus: {context_summary.get('focus', 'General')}")

    initial_context = "\n".join(context_parts)

    response, tool_call_str = await collaborator.run_decider_centric_flow(
        final_system_prompt,
        st.session_state.conversation_history,
        initial_context, 
        st.session_state.selected_model,
        metadata_info=metadata_info,
        fast_mode=False,
        thread_safe_state=thread_safe_state,
        graph_planner_advice=strategic_advice
    )


    if response:
        st.session_state.conversation_history.append({"role": "model", "parts": [response]})

    if tool_call_str:
        tool_calls = st.session_state.agent.parse_tool_calls(tool_call_str)
        if tool_calls:
            # --- Graph Memory Update: Capture State Before ---
            try:
                import rl_utils
                # Use thread_safe_state if possible as it's the active state in this scope
                state_before = rl_utils.encode_state(thread_safe_state)
            except Exception as e:
                log(f"[Graph Memory] Error encoding state before: {e}")
                # Fallback to empty/zeros if critical? 
                # Ideally we skip update if this fails, but let's try to proceed
                state_before = None

            tool_results = await collaborator.launch_parallel_tools(tool_calls, thread_safe_state, int(st.session_state.max_concurrent_gpu))
            
            # --- Graph Memory Update: Capture State After & Update ---
            if state_before is not None:
                try:
                    state_after = rl_utils.encode_state(thread_safe_state)
                    
                    # Match results to calls
                    for i, tool_result in enumerate(tool_results):
                        # Find corresponding tool name
                        tool_name = "unknown"
                        tool_args = {}
                        if i < len(tool_calls):
                            # Handle different parsing variants - Key is 'tool_name' in agent.py
                            tool_name = tool_calls[i].get("tool_name", tool_calls[i].get("tool", tool_calls[i].get("name", "tool_call")))
                            tool_args = tool_calls[i].get("args", {})
                        
                        rl_utils.update_graph_memory(thread_safe_state, tool_name, tool_result, state_before, state_after, tool_args=tool_args)
                except Exception as e:
                    log(f"[Graph Memory Update Loop ERROR] {e}")

            for tool_result in tool_results:
                st.session_state.conversation_history.append({"role": "user", "parts": [f"<tool_result>\n{json.dumps(tool_result)}\n</tool_result>"]})

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    init_log()
    log("--- Starting Headless Agent ---")

    # Initialize state
    st.session_state.cfg = cfg
    st.session_state.conversation_history = []
    st.session_state.results_log = []
    st.session_state.best_accuracy = 0.0
    st.session_state.best_params = {}
    st.session_state.global_step_counter = 0
    # st.session_state.selected_model = cfg.optimization.selected_model # REMOVED: Managed by load_state and fallback
    st.session_state.optimization_target = cfg.optimization.target  # Load target from config
    st.session_state.raw_data_source = cfg.data.raw_data_source    # Dataset ID for graph state encoding
    # --- GPU CLAMPING FIX (vLLM / Isolation Support) ---
    # Detect visible GPUs (e.g. if vLLM took some, we only see the remainder)
    visible_gpus = len(tf.config.list_physical_devices('GPU'))
    requested_gpus = cfg.execution.max_concurrent_gpu
    
    if visible_gpus > 0:
        if requested_gpus > visible_gpus:
            log(f"[Init] WARNING: Config requested {requested_gpus} GPUs, but only {visible_gpus} are visible/available.")
            log(f"[Init] Clamping max_concurrent_gpu to {visible_gpus} to prevent OOM/Index Errors.")
            st.session_state.max_concurrent_gpu = visible_gpus
        else:
            st.session_state.max_concurrent_gpu = requested_gpus
            log(f"[Init] GPU Config OK: Requested {requested_gpus}, Visible {visible_gpus}.")
    else:
        log(f"[Init] WARNING: No GPUs detected by TensorFlow! Defaulting to Config: {requested_gpus} (Might fail if tasks require GPU).")
        st.session_state.max_concurrent_gpu = requested_gpus # Fallback
    # ---------------------------------------------------
    st.session_state.manual_chat_mode = cfg.execution.manual_chat_mode
    st.session_state.memory_reset_interval = cfg.execution.memory_reset_interval

    # Purge any stale user_message.txt left over from a previous crashed session
    # so it cannot be replayed as a new instruction at startup.
    _stale_msg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_message.txt")
    if os.path.exists(_stale_msg):
        try:
            os.remove(_stale_msg)
            log("[Init] Purged stale user_message.txt from previous session.")
        except OSError as _e:
            log(f"[Init] WARNING: Could not purge stale user_message.txt: {_e}")
    st.session_state.strategic_log = []
    st.session_state.architecture_flags = {}
    st.session_state.selected_raw_data_source = cfg.data.get("raw_data_source", None)
    
    # --- STRATEGIC ADVISOR CONFIGURATION (MOSAN) ---
    # Load settings with safe defaults (Migration from 'graph_planner' to 'strategic_advisor')
    strategic_cfg = cfg.get("strategic_advisor", {})
    # Fallback to legacy key if new one is missing
    if not strategic_cfg:
        strategic_cfg = cfg.get("graph_planner", {})

    st.session_state.strategic_advisor_enabled = strategic_cfg.get("enabled", False)
    st.session_state.strategic_default_lens = strategic_cfg.get("default_lens", "auto")
    st.session_state.graph_use_meta_rules = strategic_cfg.get("use_meta_rules", True)
    st.session_state.graph_maintenance_interval = strategic_cfg.get("maintenance_interval", 10)
    st.session_state.hybrid_advice_interval = strategic_cfg.get("hybrid_advice_interval", 5)
    
    log(f"[Strategic Advisor] Enabled: {st.session_state.strategic_advisor_enabled}")
    if st.session_state.strategic_advisor_enabled:
        log(f"[Strategic Advisor] Default Lens: {st.session_state.strategic_default_lens}")
    # ---------------------------------------

    # Initialize memory with SQLite backend (fallback to JSON if fails)
    from memory_utils import create_vector_memory
    st.session_state.memory = create_vector_memory(backend="sqlite", fallback_to_json=True)
    load_state(st.session_state)

    # --- NEW EXPERIMENT DETECTION ---
    # If the dataset in config differs from the last saved run, reset trial state.
    current_data_source = cfg.data.get("raw_data_source", "")
    saved_data_source = st.session_state.get("_saved_raw_data_source", "")
    if saved_data_source and current_data_source and current_data_source != saved_data_source:
        log(f"[New Experiment] Dataset changed: '{saved_data_source}' → '{current_data_source}'. Resetting trial state.")
        st.session_state.results_log = []
        st.session_state.best_accuracy = 0.0
        st.session_state.best_params = {}
        st.session_state.global_step_counter = 0
        st.session_state.architecture_flags = {}
    # --------------------------------

    # Fallback: If state didn't provide a selected_model, use config
    if "selected_model" not in st.session_state or not st.session_state.selected_model:
        st.session_state.selected_model = cfg.optimization.selected_model
        log(f"[Main] State did not contain selected_model. Initialized from config: {st.session_state.selected_model}")

    # --- VERIFICATION LOGGING (User Request) ---
    current_source = st.session_state.get("selected_raw_data_source")
    log(f"[Main] ACTIVE RAW DATA SOURCE: '{current_source}' (Loaded from Config: {cfg.data.get('raw_data_source')})")
    # -------------------------------------------



    # --- HITL RESUME LOGIC ---
    hitl_marker = os.path.join("cluster", "hitl_resume_marker.json")
    hitl_checkpoint = os.path.join("cluster", "hitl_checkpoint.json")
    
    if os.path.exists(hitl_marker):
        log("[HITL] 🟢 Resume Signal Detected. Attempting to restore state...")
        if os.path.exists(hitl_checkpoint):
            try:
                with open(hitl_checkpoint, "r") as f:
                    data = json.load(f)
                    st.session_state.conversation_history = data.get("conversation_history", [])
                    st.session_state.results_log = data.get("results_log", [])
                    st.session_state.best_accuracy = data.get("best_accuracy", 0.0)
                    st.session_state.best_params = data.get("best_params", {})
                    st.session_state.global_step_counter = data.get("global_step_counter", 0)
                    st.session_state.selected_model = data.get("selected_model", "UNKNOWN_MODEL")
                    st.session_state.architecture_flags = data.get("architecture_flags", {})
                log("[HITL] State successfully restored.")
            except Exception as e:
                log(f"[HITL] ❌ Failed to load checkpoint: {e}")
        
        # Check for user feedback
        feedback_file = os.path.join("cluster", "approval_feedback.json")
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, "r") as f:
                    fb = json.load(f)
                    msg = fb.get("message", "Tool approved.")
                    tool_name = fb.get("tool_name", "Unknown Tool")
                
                # INJECT FEEDBACK INTO HISTORY
                feedback_entry = {
                    "role": "user", 
                    "parts": [f"IMPORTANT SYSTEM NOTICE: The pending tool '{tool_name}' has been APPROVED by the user.\n\nUser Feedback/Instruction:\n\"{msg}\"\n\nYou may now proceed with using this tool."]
                }
                st.session_state.conversation_history.append(feedback_entry)
                log(f"[HITL] User feedback injected: {msg}")
                
                # Cleanup feedback file
                os.remove(feedback_file)
            except Exception as e:
                log(f"[HITL] Error reading feedback: {e}")

        # Cleanup Marker (consume the resume signal)
        os.remove(hitl_marker)
    # -------------------------
    update_best_result_from_log()
    reinitialize_agent(cfg)
    _dataset_id = st.session_state.get("selected_raw_data_source", cfg.data.raw_data_source) or "global"
    _ghost_threshold = cfg.strategic_advisor.get("ghost_cold_start_threshold", 10)
    st.session_state.strategic_graph_memory = StrategicGraphMemory(
        dataset_id=_dataset_id,
        ghost_threshold=_ghost_threshold,
    )
    st.session_state.strategic_graph_memory.save_mosan_metadata(
        representation_type=cfg.optimization.get("selected_model", "1D_CNN"),
        data_domain="SERS_spectral",
    )

    log("--- Initialization Complete ---")

    # Construct thread_safe_state using the SharedStateManager Proxy
    # This ensures that any change to 'thread_safe_state' inside agents is immediately reflected in 'st.session_state'
    thread_safe_state = SharedStateManager(st.session_state)

    # We no longer need to manually copy fields, as the proxy forwards everything.
    # But for backward compatibility if any tool relies on keys being present immediately:
    # (The proxy forwards .get() calls to st.session_state, so they ARE present if they are in session_state)

    # Ensure critical keys exist in session_state
    if "results_log" not in st.session_state: st.session_state.results_log = []
    if "conversation_history" not in st.session_state: st.session_state.conversation_history = []
    if "strategic_graph_memory" not in st.session_state:
        st.session_state.strategic_graph_memory = StrategicGraphMemory(
            dataset_id=_dataset_id,
            ghost_threshold=_ghost_threshold,
        )
        st.session_state.strategic_graph_memory.save_mosan_metadata(
            representation_type=cfg.optimization.get("selected_model", "1D_CNN"),
            data_domain="SERS_spectral",
        )


    async def run_steps():
        # thread_safe_state is now available from the outer scope

        # Define recursive execution wrapper
        def recursive_execute_tool(name, args, st_state, gpu_id=None):
             return execute_tool(name, args, st_state, gpu_id, execute_tool_func=recursive_execute_tool)

        num_gpus_available = len(tf.config.list_physical_devices('GPU'))
        log(f"[Main] {num_gpus_available} GPUs detected (SLURM handled isolation). Using all available.")

        # Initialize DeciderOrchestrator ONCE
        collaborator = DeciderOrchestrator(
            st.session_state.agent, 
            recursive_execute_tool,
            num_gpus=num_gpus_available
        )

        # Import needed for the context manager
        from cluster.mlflow_utils import MLFlowAgentContext, log_agent_artifact

        # Session-start: build the Research Chronicle ONCE (standalone technical pull-based report)
        # and flag it for the FIRST Theorist cycle. Thereafter it is pull-only via the
        # refresh_chronicle tool — NOT regenerated every step.
        try:
            from summarizer_ledger import build_research_chronicle
            if build_research_chronicle(thread_safe_state):
                thread_safe_state["chronicle_deliver_next"] = True
                log("[Main] Research Chronicle bootstrapped at session start.")
        except Exception as e:
            log(f"[Main] Chronicle bootstrap skipped: {e}")

        for i in range(cfg.execution.num_steps):
            log(f"--- Running Step {i+1}/{cfg.execution.num_steps} ---")

            # --- MLFLOW PARENT RUN START ---
            step_run_name = f"Turn_{st.session_state.global_step_counter + 1}" # +1 because counter increments inside
            with MLFlowAgentContext(experiment_name=cfg.logging.mlflow.experiment_name, run_name=step_run_name) as parent_run:
                # Capture the run_id to pass down (if needed, though currently we rely on active/context)
                # Ideally we pass this run_id to the agents so they know who their parent is.
                # The current `run_agent_turn` (and decider) doesn't explicitly take run_id in the signature above, 
                # but `collaborator.run_decider_centric_flow` does create tasks.
                # We can store the current run_id in session_state for global access or modify signatures.
                # For least intrusion, let's inject it into thread_safe_state "context"
                
                current_run_id = parent_run.info.run_id if parent_run else None
                if current_run_id:
                    st.session_state.current_mlflow_run_id = current_run_id
                    thread_safe_state["current_mlflow_run_id"] = current_run_id
                    log(f"[MLFlow] Started Parent Run: {step_run_name} (ID: {current_run_id})")

                # --- DYNAMIC TOOL RE-LOADING (Start of Step) ---
                # Reload tools at every step or just at start, to catch newly approved items if restart happens.
                # Ideally done once at start of 'main' but doing here ensures robustness.
                loader = DynamicToolLoader()
                new_tools = loader.load_tools()
                if new_tools:
                    AVAILABLE_TOOLS.update(new_tools)
                    # log(f"[Dynamic Loader] Updated AVAILABLE_TOOLS with: {list(new_tools.keys())}")
                # -----------------------------------------------

                # --- Strategic Graph Evaluation (Periodic) ---
                graph_eval_interval = cfg.execution.get("graph_evaluation_interval", 10)
                if i > 0 and i % graph_eval_interval == 0:
                    log("[Strategic Graph Evaluator] Starting periodic evaluation...")
                    try:
                        # Import here to avoid circular deps if any
                        from agent import StrategicGraphEvaluatorCollaborator
                        graph_collaborator = StrategicGraphEvaluatorCollaborator(st.session_state.agent)
                        # We run this synchronously for now to ensure the insight is available for the turn
                        graph_insight = await graph_collaborator.run_collaboration()
                        st.session_state.latest_strategic_insight = graph_insight.get("report", "No report generated.")
                        log("[Strategic Graph Evaluator] Insight generated and stored.")
                    except Exception as e:
                        log(f"[Strategic Graph Evaluator] Failed: {e}")

                # --- SYNC STATE (Critical Fix) ---
                # REPLACED BY SHARED STATE MANAGER PROXY
                # The SharedStateManager ensures that 'thread_safe_state' is a live proxy to 'st.session_state'
                # So manual synchronization is no longer needed.
                # ---------------------------------

                # --- OLD ORCHESTRATION (RETIRED — Phase 3 cutover 2026-03-19) ---
                # The old run_agent_turn + DeciderOrchestrator has been replaced by
                # the Single Ledger pipeline: Analyst → Theorist → Decider.
                # Rollback: uncomment the block below and comment out the ledger phases.
                #
                # try:
                #     await run_agent_turn(thread_safe_state, collaborator)
                # except Exception as e:
                #     log(f"[Main Loop Error] Step {i+1} failed with error: {e}")
                # finally:
                #     save_state(st.session_state)
                # ----------------------------------------------------------------

                # --- SINGLE LEDGER PIPELINE (replaces run_agent_turn) ---
                
                # Check user directive before proceeding
                directive = read_user_directive("workspace/")
                
                while directive.status == "paused":
                    log("[Single Ledger] System paused by user directive. Waiting...")
                    await asyncio.sleep(5)
                    directive = read_user_directive("workspace/")
                    if directive.status != "paused":
                        log(f"[Single Ledger] Resuming with status: {directive.status}")
                        break
                        
                if directive.status == "stopped":
                    log("[Single Ledger] System stopped by user directive.")
                    break

                st.session_state.global_step_counter += 1
                log(f"--- Starting Global Step: {st.session_state.global_step_counter} ---")

                # Discover previous ledger before creating the new one
                prev_cycle_num, prev_ledger_path = get_latest_ledger()
                next_cycle_num = prev_cycle_num + 1

                # Check epoch limits
                if directive.max_epochs and next_cycle_num > directive.max_epochs:
                    log(f"[Single Ledger] Max epochs ({directive.max_epochs}) reached. Stopping.")
                    break
            
                if directive.pause_after_epoch and next_cycle_num > directive.pause_after_epoch:
                    log(f"[Single Ledger] Pause point (epoch {directive.pause_after_epoch}) reached. Stopping.")
                    break

                # Phase 0 chronicle removed: the Research Chronicle is now a standalone pull-based
                # technical report (workspace/research_chronicle.md) — generated ONCE at session
                # start (above) and on Theorist request via refresh_chronicle, NOT injected
                # per-cycle as Part 0.

                # Phase 1: Analyst Post-Mortem
                next_ledger_path = None
                try:
                    next_ledger_path = create_ledger(next_cycle_num, preamble=None)
                    run_analyst(
                        ledger_path=next_ledger_path,
                        thread_safe_state=thread_safe_state,
                        prev_ledger_path=prev_ledger_path,
                        experiment_name=cfg.logging.mlflow.experiment_name,
                        user_notes=directive.notes,
                    )
                except Exception as e:
                    log(f"[Single Ledger] Analyst phase failed: {e}")
                    next_ledger_path = None

                # Phase 2: Theorist Deep Searchu
                try:
                    if next_ledger_path:
                        theorist_ok = run_theorist(
                            ledger_path=next_ledger_path,
                            thread_safe_state=thread_safe_state,
                            priority_override=directive.priority_override,
                            constraints=directive.constraints,
                            notes=directive.notes,
                        )
                        log(f"[Single Ledger] Theorist {'completed' if theorist_ok else 'halted'} — check ledger.")
                except Exception as e:
                    log(f"[Single Ledger] Theorist phase failed: {e}")

                # Phase 3: Decider Execution — checkpoint-based (runs if Part 3 exists)
                try:
                    if next_ledger_path and SECTION_MARKERS["part_3"] in read_ledger(next_ledger_path):
                        decider_success = run_decider(
                            ledger_path=next_ledger_path,
                            thread_safe_state=thread_safe_state,
                            execute_tool_func=execute_tool,
                            constraints=directive.constraints,
                            prev_ledger_path=prev_ledger_path,
                        )
                        log(f"[Single Ledger] Decider {'completed — Part 4 written' if decider_success else 'failed'}.")
                    else:
                        log("[Single Ledger] Part 3 not found — Decider skipped.")
                except Exception as e:
                    log(f"[Single Ledger] Decider phase failed: {e}")

                # Phase 4: Summarizer — distill cycle into ledger_appendix_{N}.md
                try:
                    if next_ledger_path:
                        run_multi_agent_summarizer(
                            ledger_path=next_ledger_path,
                            thread_safe_state=thread_safe_state,
                        )
                except Exception as e:
                    log(f"[Single Ledger] Summarizer phase failed: {e}")

                # --- MOSAN: Periodic Hybrid Advice (every N ledger cycles, toggle-gated) ---
                if (
                    st.session_state.get("strategic_advisor_enabled", False)
                    and next_cycle_num is not None
                    and next_cycle_num > 0
                    and next_cycle_num % st.session_state.get("hybrid_advice_interval", 5) == 0
                ):
                    log(f"[MOSAN] Running HybridGraphAdvisorCollaborator (cycle {next_cycle_num})...")
                    try:
                        from tools import get_hybrid_advice
                        advice_result = get_hybrid_advice(thread_safe_state=thread_safe_state)
                        if advice_result.get("status") == "completed":
                            thread_safe_state["latest_hybrid_advice"] = advice_result.get("report", "")
                            log("[MOSAN] Hybrid advice stored for next Knowledge Scout.")
                        else:
                            log(f"[MOSAN] Hybrid advice failed: {advice_result.get('message')}")
                    except Exception as e:
                        log(f"[MOSAN] Hybrid advice error: {e}")
                # -----------------------------------------------------------------------

                # Save state AFTER all ledger phases (captures Decider results)
                save_state(st.session_state)
                # ------------------------------------------------

                # --- MOSAN: Periodic Summarizer (every memory_reset_interval ledgers, toggle-gated) ---
                if st.session_state.get("strategic_advisor_enabled", False):
                    reset_interval = st.session_state.get("memory_reset_interval", 20)
                    if next_cycle_num is not None and next_cycle_num > 0 and next_cycle_num % reset_interval == 0:
                        log(f"[MOSAN] Triggering strategic summarizer (cycle {next_cycle_num})...")
                        try:
                            trigger_strategic_summary_and_reset_memory(
                                thread_safe_state,
                                reset_conversation=False,  # Single Ledger is air-gapped — no conversation to reset
                            )
                        except Exception as e:
                            log(f"[MOSAN] Summarizer failed: {e}")
                # ---------------------------------------------------------------------------------------

                # --- RESULTS LOG SNAPSHOT ---
                if current_run_id and st.session_state.results_log:
                    try:
                        snapshot = json.dumps(st.session_state.results_log, indent=2, default=str)
                        log_agent_artifact(current_run_id, "results_log_snapshot.json", snapshot, artifact_path="state")
                        log(f"[MLFlow] Saved results_log snapshot to run {current_run_id}")
                    except Exception as e:
                        log(f"[MLFlow] Warning: Failed to save results snapshot: {e}")

                # Force garbage collection
                import gc
                gc.collect()

        # --- POST-RUN SUMMARIZATION & IDLE WAIT MODE ---
        log("--- Autonomous Steps Completed ---")
        
        # 1. Post-Run Summarization
        # We trigger a final summary so the next start has a clear context.
        # Check if we didn't just summarize on the very last step naturally
        reset_interval = st.session_state.get("memory_reset_interval", 20)
        if cfg.execution.num_steps % reset_interval != 0 and st.session_state.global_step_counter > 0:
            log("[Main] Triggering final post-run summary to distill context...")
            trigger_strategic_summary_and_reset_memory(thread_safe_state, special_pretext="Final summarization after autonomous steps completed.")

        # 2. Idle Wait Mode
        if st.session_state.get("manual_chat_mode", False):
            log(f"--- Entering Idle Wait Mode (manual_chat_mode = True) ---")
            log("[Main] Waiting for 'user_message.txt' or manual intervention. Press Ctrl+C to exit.")
            # Sync flag into thread_safe_state so _check_for_user_message can see it
            thread_safe_state["manual_chat_mode"] = True
            
            idle_check_interval = 5  # seconds
            
            try:
                while True:
                    # Look for the user message file
                    user_msg = collaborator._check_for_user_message(thread_safe_state)
                    
                    if user_msg:
                         log("[Main] User message detected in Idle Mode! Processing one extra turn...")
                         # Inject the user message
                         st.session_state.conversation_history.append({"role": "user", "parts": [user_msg]})
                         
                         # Run one extra turn
                         try:
                             # We need the MLFlow context for the extra turn
                             extra_run_name = f"Turn_{st.session_state.global_step_counter + 1}_Interactive"
                             from cluster.mlflow_utils import MLFlowAgentContext
                             with MLFlowAgentContext(experiment_name=cfg.logging.mlflow.experiment_name, run_name=extra_run_name) as extra_run:
                                 if extra_run:
                                     st.session_state.current_mlflow_run_id = extra_run.info.run_id
                                 await run_agent_turn(thread_safe_state, collaborator)
                             
                             log("[Main] Extra turn completed. Returning to Idle Wait Mode.")
                             log("[Main] Waiting for 'user_message.txt'...")
                         except Exception as e:
                             log(f"[Main] Error processing interactive turn: {e}")
                             traceback.print_exc()
                         finally:
                             save_state(st.session_state)
                             import gc
                             gc.collect()
                    else:
                        # Sleep briefly to prevent high CPU usage
                        await asyncio.sleep(idle_check_interval)
                        
            except KeyboardInterrupt:
                log("[Main] Idle loop interrupted by user.")
        else:
             log("--- Exiting (manual_chat_mode is False) ---")
        # -----------------------------------------------

    if cfg.execution.get("mode", "optimization") == "report":

        log("--- REPORT GENERATION MODE ---")
        report_source = cfg.execution.get("report_source", "processed_data/logs/report_staging")
        
        # Resolve path relative to project root if needed
        if not os.path.isabs(report_source):
            report_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), report_source)

        combined_history = []
        
        if os.path.isdir(report_source):
            log(f"Loading conversation archives from folder: {report_source}")
            if not os.path.exists(report_source):
                os.makedirs(report_source, exist_ok=True)
                log(f"Created staging folder at {report_source}. Please place JSON files here.")
                return

            json_files = [f for f in os.listdir(report_source) if f.endswith(".json")]
            if not json_files:
                log("No JSON files found in the staging folder.")
                return
            
            for f_name in json_files:
                f_path = os.path.join(report_source, f_name)
                try:
                    with open(f_path, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            combined_history.extend(data)
                            log(f"Loaded {len(data)} items from {f_name}")
                except Exception as e:
                    log(f"Error loading {f_name}: {e}")
        
        elif os.path.isfile(report_source):
            log(f"Loading conversation archive from file: {report_source}")
            try:
                with open(report_source, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        combined_history = data
                        log(f"Loaded {len(data)} items.")
            except Exception as e:
                log(f"Error loading file: {e}")
                return
        else:
            log(f"Invalid report source: {report_source}")
            return

        # Store combined history in state for the tool to access
        st.session_state.conversation_history = combined_history
        st.session_state.report_data = combined_history # Explicit field for clarity
        
        # CRITICAL: Update thread_safe_state with the loaded data!
        thread_safe_state["conversation_history"] = combined_history
        thread_safe_state["report_data"] = combined_history
        
        # Define recursive execution wrapper for report mode
        def recursive_execute_tool(name, args, st_state, gpu_id=None):
             return execute_tool(name, args, st_state, gpu_id, execute_tool_func=recursive_execute_tool)

        # Trigger the report generation tool
        log("Triggering Scientific Report Generation...")
        # We pass dummy values for dataset/architecture as they might be inferred or not relevant for a meta-report
        execute_tool(
            "generate_scientific_report", 
            {"dataset": "combined_sessions", "architecture": "various"}, 
            thread_safe_state,
            None, # gpu_id
            recursive_execute_tool # execute_tool_func
        )
        log("--- Report Generation Complete ---")
        return

    elif cfg.execution.get("mode", "optimization") == "imagenet_competition":
        # --- IMAGENET COMPETITION MODE ---
        # Lazy import: the imagenet module is NEVER loaded in optimization/report mode.
        # This guarantees zero disruption to the existing workflow.
        imagenet_cfg = cfg.get("imagenet", {})
        if not imagenet_cfg.get("enabled", False):
            log("[ImageNet] ⚠️  Mode is 'imagenet_competition' but imagenet.enabled=false in config.yaml.")
            log("[ImageNet] Set 'imagenet.enabled: true' to activate. Exiting.")
            return
        log("--- IMAGENET COMPETITION MODE ---")
        from imagenet.competition_runner import run_imagenet_campaign
        run_imagenet_campaign(cfg, thread_safe_state)
        log("--- ImageNet Campaign Complete ---")
        return

    # --- OPTIMIZATION MODE (Default) ---
    asyncio.run(run_steps())
    log("--- Headless Agent Finished ---")

if __name__ == "__main__":
    main()
