import os
import json
import re
import asyncio
import mlflow
from omegaconf import OmegaConf
from typing import Dict, Any, List

from .celery_app import app
from .mlflow_utils import log_agent_artifact
from tools import AVAILABLE_TOOLS
from ui_logger import log
from .trace_logger import save_agent_trace
import time

def wait_for_run_propagation(run_id: str, max_retries: int = 5, retry_delay: float = 1.0):
    """
    Waits for the MLFlow Run to be visible in the tracking store.
    Useful for distributed filesystems with latency.
    """
    for attempt in range(max_retries):
        try:
            mlflow.get_run(run_id)
            return True
        except Exception:
            log(f"[MLFlow-Worker] Run {run_id} not found yet. Retrying {attempt+1}/{max_retries}...")
            time.sleep(retry_delay)
    
    log(f"[MLFlow-Worker] WARNING: Run {run_id} could not be found after {max_retries} attempts. Logging might fail.")
    return False

def _rehydrate_agent(model_override: str = None):
    """
    Creates a new instance of ConversationalAgent using config.yaml.
    This runs inside the Celery Worker process.
    """
    # CRITICAL: Set tracking URI immediately to avoid 'Experiment 0' issues during lazy imports
    try:
        from .mlflow_utils import MLFLOW_TRACKING_URI as _canonical_uri, bootstrap_mlflow
        bootstrap_mlflow()
        mlflow.set_tracking_uri(_canonical_uri)
    except Exception as e:
        log(f"[Worker] Warning: Could not set MLflow URI in _rehydrate_agent: {e}")

    # Import locally to avoid top-level side effects
    try:
        from agent import ConversationalAgent
    except ImportError:
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from agent import ConversationalAgent

    # Locate config.yaml (assuming it's in the project root)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.join(project_root, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # Load config via OmegaConf
    try:
        cfg = OmegaConf.load(config_path)
    except Exception as e:
        raise ValueError(f"Failed to load config.yaml: {e}")

    # Standardized URI resolution
    from .mlflow_utils import MLFLOW_TRACKING_URI as _canonical_uri
    mlflow.set_tracking_uri(_canonical_uri)

    # Extract API details based on provider
    provider = cfg.api.get("provider", "Openrouter").lower()
    
    if provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
        api_key = cfg.api.get("openrouter_api_key")
        model_name = cfg.api.get("openrouter_model_name")
    elif "local" in provider or "lm studio" in provider: 
        base_url = "http://host.docker.internal:1234/v1"
        api_key = "not-needed"
        model_name = "local-model"
    elif "vllm" in provider:
        base_url = cfg.api.get("local_llm_url", "http://localhost:8000/v1")
        api_key = "EMPTY"
        # Dynamically fetch model name from vLLM (same logic as main.py)
        model_name = "local-model"
        try:
            import requests as _req
            resp = _req.get(f"{base_url}/models", timeout=5)
            if resp.status_code == 200:
                model_name = resp.json()["data"][0]["id"]
                log(f"[Worker] Detected vLLM model: {model_name}")
        except Exception as e:
            log(f"[Worker] Warning: Could not detect vLLM model: {e}. Using 'local-model'.")
    else:
        # Fallback to env or unknown
        raise ValueError(f"Unsupported provider in worker: {provider}")

    if not api_key:
        raise ValueError(f"API Key missing for provider {provider} in config.yaml")

    # CRITICAL FIX: Use the dynamic model name passed from main process if available
    if model_override:
        model_name = model_override

    return ConversationalAgent(base_url=base_url, api_key=api_key, model_name=model_name)

@app.task
def run_analyst_task(initial_context: str, thread_safe_state: Dict[str, Any], run_id: str, model_name: str = None, tracking_uri: str = None):
    """
    Celery Task for the Analyst Agent.
    Executes fast analysis tools and summarizes facts.
    """
    # Enforce Tracking URI consistency
    from .mlflow_utils import MLFLOW_TRACKING_URI as _canonical_uri, bootstrap_mlflow
    if tracking_uri:
        bootstrap_mlflow(tracking_uri)
    else:
        bootstrap_mlflow()
        mlflow.set_tracking_uri(_canonical_uri)
    
    # Wait for run to appear
    run_exists = wait_for_run_propagation(run_id)

    agent = _rehydrate_agent(model_override=model_name)
    log(f"[Analyst-Worker] Starting task for Run ID: {run_id} (Model: {agent.model_name})")

    # --- 1. Automated Data Gathering (SMART CONTEXT) ---
    current_model = thread_safe_state.get("selected_model")
    
    # Define distinct data requests
    # Request A: Recent Trials (Immediate context)
    req_recent = {"tool_name": "get_recent_trials", "args": {"n": 5}}
    
    # Request B: Best Trials (Historical "Ceiling" Context)
    req_best = {"tool_name": "get_sorted_trials", "args": {"sort_by": "mean_accuracy", "ascending": False, "n": 5}}
    
    # Request C: Deep Analysis
    req_analysis = [
        {"tool_name": "analyze_best_vs_worst_trials", "args": {}},
        {"tool_name": "get_optimization_status", "args": {}},
        {"tool_name": "get_correlation_matrix", "args": {}} 
    ]
    
    analysis_tools_requests = [req_recent, req_best] + req_analysis
    
    gathered_data_dict = {} # Map tool_name -> result string
    
    for tool_req in analysis_tools_requests:
        tool_name = tool_req["tool_name"]
        args = tool_req["args"].copy()
        
        # Smart Filter Injection
        if "architecture_filter" not in args or args["architecture_filter"] is None:
            if current_model:
                args["architecture_filter"] = current_model
        
        try:
            if tool_name in AVAILABLE_TOOLS:
                tool_func = AVAILABLE_TOOLS[tool_name]
                try:
                    result = tool_func(thread_safe_state=thread_safe_state, **args)
                except TypeError as e:
                    if "unexpected keyword argument" in str(e) and "architecture_filter" in args:
                        args.pop("architecture_filter")
                        result = tool_func(thread_safe_state=thread_safe_state, **args)
                    else:
                        raise e
                
                # Store raw result (for de-duplication if needed later)
                filter_used = args.get('architecture_filter', 'Global')
                key = f"{tool_name}_{filter_used}"
                gathered_data_dict[key] = f"--- {tool_name} Results ({filter_used}) ---\n{json.dumps(result, indent=2)}"
            else:
                 gathered_data_dict[tool_name] = f"--- {tool_name} Error ---\nTool not found in worker registry."

        except Exception as e:
            gathered_data_dict[tool_name] = f"--- {tool_name} Error ---\n{str(e)}"

    # Construct Full Context
    # We simply concatenate for now, as direct de-duplication of complex JSON strings is risky/expensive.
    # The overlapping info between "Recent" and "Best" is acceptable redundancy.
    full_context = f"{initial_context}\n\n" + "\n\n".join(gathered_data_dict.values())

    # --- 2. LLM Summary ---
    analyst_system_prompt = """
    You are part of the Strategy Team, you are a meticulous Data Analyst. Your sole task is to summarize the objective facts based on the provided data context.
     
     **SCIENTIFIC REVIEWER GUIDELINES (STRICT):**
     1. **Flag Overfitting:** If the context shows a gap between Train and Test accuracy > 5% (0.05), you MUST explicitly flag "High Overfitting Risk".
     2. **Correlation != Causation:** Do NOT claim "X causes Y" based on correlation alone. Say "associated with".
     3. **Ignore Noise:** If a correlation is weak (< 0.2) or labeled "insignificant" (p > 0.05), IGNORE IT completely. Do not report it.
     4. **Discrete Architecture:** When reporting Kernels, Filters, or Layers, report the MODE (integer), never a float (e.g. say "Kernel Size 3", NOT "3.67").
     5. **Missing Data:** If variables like `representation_type` are NULL, state "Missing critical metadata".

     **YOUR GOAL: DETAILED REPORTING**
     - DO NOT be concise. We need DETAILS.
     - explicitly flag REPETITIONS (e.g. "We have run the same trial 5 times").
     - explicitly flag FAILURES or STALLED METRICS.
     - DO NOT make hypotheses, suggestions, or action plans.
     - DO NOT mention any tools or coding.
    """
    
    try:
        # Sync call inside worker is fine
        analysis, _ = agent.run_turn(
            analyst_system_prompt,
            [{"role": "user", "parts": [f"Here is the data context for {current_model}:\n{full_context}"]}]
        )
        
        # --- 3. MLFlow Logging (Child Run) ---
        if run_exists:
            try:
                # Resume parent run solely to nest the child run under it
                with mlflow.start_run(run_id=run_id):
                    with mlflow.start_run(run_name="Analyst", nested=True):
                        mlflow.log_text(analysis, "output/analyst_report.md")
                        mlflow.log_text(full_context, "debug/full_context.txt")
                        log(f"[Analyst-Worker] Final Analysis logged to MLFlow (Child Run)")
            except Exception as ml_err:
                log(f"[Analyst-Worker] Warning: MLFlow logging failed (Run might be missing): {ml_err}")

        log(f"[Analyst-Worker] Final Analysis:\n{analysis}")
        
        # --- 4. Detailed Trace Logging (For Fine-Tuning) ---
        save_agent_trace(
            run_id=run_id,
            agent_name="Analyst",
            system_prompt=analyst_system_prompt,
            inputs=full_context,
            output=analysis,
            metadata={"model": agent.model_name, "raw_data_sources": [t["tool_name"] for t in analysis_tools_requests]}
        )
        
        # Log gathered data for debugging too
        log_agent_artifact(run_id, "analyst_raw_data.txt", full_context, artifact_path="debug")
        
        return analysis

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        err_msg = f"Analyst failed: {str(e)}\n\nTraceback:\n{tb}"
        log_agent_artifact(run_id, "analyst_error.txt", err_msg)
        return err_msg

@app.task
def run_strategist_task(conversation_history: List[Dict], run_id: str, model_name: str = None, analyst_report: str = None, graph_advice: str = None, current_todos: List[Dict] = None, tracking_uri: str = None):
    """
    Celery Task for the Strategist Agent.
    Filters history and generates strategic hypotheses.
    Now accepts 'analyst_report' for sequential intelligence.
    Now accepts 'graph_advice' for long-term memory guidance.
    Now accepts 'current_todos' to align with Decider's active plan.
    """
    # Enforce Tracking URI consistency
    from .mlflow_utils import MLFLOW_TRACKING_URI as _canonical_uri, bootstrap_mlflow
    if tracking_uri:
        bootstrap_mlflow(tracking_uri)
    else:
        bootstrap_mlflow()
        mlflow.set_tracking_uri(_canonical_uri)
        
    # Wait for run to appear
    run_exists = wait_for_run_propagation(run_id)

    agent = _rehydrate_agent(model_override=model_name)
    log(f"[Strategist-Worker] Starting task for Run ID: {run_id} (Model: {agent.model_name})")
    
    # --- 1. History Filtering (Simplified Logic for Worker) ---
    filtered_history = []
    
    # We implement a safer, simpler filter here to avoid regex complexity issues in re-implementation
    # or copying the massive regex block.
    # Key Logic: 
    # - User 'Analyst Summary' -> Keep
    # - User 'Tool Result' (Key tools) -> Keep
    # - Assistant (Sanitized) -> Keep
    
    key_feedback_tools = [
        "run_training_trial", "run_strategic_optuna_sweep", "run_supernet_search",
        "construct_model_from_genotype", "get_optimization_status", "run_web_research",
        "generate_creative_hypotheses", "run_architecture_comparison",
        "propose_intelligent_architecture", "validate_architecture_file", "write_architecture_file"
    ]

    for i, msg in enumerate(conversation_history):
        role = msg.get("role")
        role = msg.get("role")
        parts = msg.get("parts")
        if parts and isinstance(parts, list) and len(parts) > 0:
            content = parts[0]
        else:
            content = msg.get("content", "")
        
        if role == "user":
            if "Analyst's summary" in content:
                filtered_history.append({"role": "user", "content": content})
            elif "<tool_result>" in content or '"status": "error"' in content or '"genotype":' in content:
                 # Heuristic inclusion for results
                 filtered_history.append({"role": "user", "content": content})
        elif role in ["assistant", "model"]:
             # Remove thought tags
             clean_content = re.sub(r"</?thought_step_\d+.*?>", "", content, flags=re.DOTALL | re.IGNORECASE)
             clean_content = re.sub(r"\((Analyst|Strategist|Decider)[^)]*\)", "", clean_content)
             if clean_content.strip():
                filtered_history.append({"role": "assistant", "content": clean_content.strip()})

    # --- 2. LLM Hypothesizing ---

    # --- ARCHITECTURE LIBRARY INJECTION ---
    # Discover available pre-built templates so the Strategist can reference them.
    library_listing = "None available."
    try:
        library_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "cluster", "library", "architectures")
        if os.path.isdir(library_path):
            arch_files = [f.replace(".py", "") for f in os.listdir(library_path) if f.endswith(".py")]
            if arch_files:
                library_listing = ", ".join(sorted(arch_files))
                log(f"[Strategist-Worker] Discovered {len(arch_files)} architecture templates: {library_listing}")
    except Exception as e:
        log(f"[Strategist-Worker] Warning: Could not read architecture library: {e}")
    # --- END INJECTION ---

    from .prompts import STRATEGIST_SYSTEM_PROMPT
    
    # Format the to-do list for the prompt
    todos_str = "No active to-do list."
    if current_todos:
        todos_str = ""
        for t in current_todos:
            status_mark = "x" if t.get("status") == "completed" else " "
            todos_str += f"- [{status_mark}] {t.get('task')}\n"

    strategist_system_prompt = STRATEGIST_SYSTEM_PROMPT.format(
        library_listing=library_listing, 
        analyst_report=analyst_report, 
        graph_advice=graph_advice,
        current_todos=todos_str.strip()
    )

    try:
        hypothesis, _ = agent.run_turn(
            strategist_system_prompt,
            filtered_history
        )
        
        # --- 3. MLFlow Logging (Child Run) ---
        if run_exists:
            try:
                with mlflow.start_run(run_id=run_id):
                    with mlflow.start_run(run_name="Strategist", nested=True):
                        conversation_history_str = json.dumps(conversation_history, indent=2, default=str)
                        mlflow.log_text(hypothesis, "output/strategist_plan.md")
                        mlflow.log_text(conversation_history_str, "debug/history_snapshot.txt")
                        log(f"[Strategist-Worker] Final Hypothesis logged to MLFlow (Child Run)")
            except Exception as ml_err:
                log(f"[Strategist-Worker] Warning: MLFlow logging failed: {ml_err}")

        log(f"[Strategist-Worker] Final Hypothesis:\n{hypothesis}")
        
        # --- 4. Detailed Trace Logging (For Fine-Tuning) ---
        save_agent_trace(
            run_id=run_id,
            agent_name="Strategist",
            system_prompt=strategist_system_prompt,
            inputs=filtered_history,
            output=hypothesis,
            metadata={"model": agent.model_name, "history_length": len(filtered_history)}
        )
        
        return hypothesis

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        err_msg = f"Strategist failed: {str(e)}\n\nTraceback:\n{tb}"
        log_agent_artifact(run_id, "strategist_error.txt", err_msg)
        return err_msg

@app.task
def run_strategic_graph_maintenance_task(run_id: str = "periodic", dataset_id: str = "global"):
    """
    Celery Task for Graph Maintenance.
    Runs the StrategicGraphEvaluatorCollaborator to refine the graph.

    Args:
        run_id: Unique identifier for this run (used for artifact logging).
        dataset_id: Active dataset identifier. Used to load/write the correct per-dataset graph file.
                    Falls back to "global" if not provided.
    """
    try:
        # --- CRITICAL FIX: Respect config.yaml strategic_advisor.enabled setting ---
        # Load config to check if strategic advisor is enabled
        from omegaconf import OmegaConf
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        config_path = os.path.join(project_root, "config.yaml")
        
        if os.path.exists(config_path):
            try:
                cfg = OmegaConf.load(config_path)
                advisor_enabled = cfg.get("strategic_advisor", {}).get("enabled", True)
                
                if not advisor_enabled:
                    log("[Graph-Worker] Strategic advisor disabled in config. Graph continues learning, skipping LLM analysis.")
                    return {
                        "status": "skipped_consultation", 
                        "reason": "strategic_advisor.enabled=false in config.yaml",
                        "note": "Graph continues growing via update_graph_memory() in main app"
                    }
            except Exception as e:
                log(f"[Graph-Worker] Warning: Could not load config.yaml: {e}. Proceeding with task.")
        # --------------------------------------------------------------------------------
        
        agent = _rehydrate_agent()
        log(f"[Graph-Worker] Starting maintenance task...")
        
        # --- FIX: Initialize Graph Memory explicitly in Worker ---
        from graph_memory import StrategicGraphMemory

        # Load the dataset-specific graph; ghost injection is skipped here (worker only maintains existing graph)
        if not dataset_id:
            log("[Graph-Worker] WARNING: dataset_id missing from task args — falling back to 'global'.")
            dataset_id = "global"
        log(f"[Graph-Worker] Loading graph for dataset_id='{dataset_id}'...")
        graph_memory = StrategicGraphMemory(dataset_id=dataset_id, ghost_threshold=0)
        log(f"[Graph-Worker] Loaded Strategic Graph with {len(graph_memory.graph.nodes)} nodes and {len(graph_memory.graph.edges)} edges.")
        
        # Create a mock thread_safe_state for tool execution
        thread_safe_state = {
            "strategic_graph_memory": graph_memory,
            "agent": agent,
            "results_log": [], # Dummy for safety
            "conversation_history": [] 
        }

        # Create a tool executor wrapper
        def worker_tool_executor(tool_name, tool_args):
            # Inject state automatically
            tool_args["thread_safe_state"] = thread_safe_state
            
            if tool_name in AVAILABLE_TOOLS:
                # AVAILABLE_TOOLS functions typically expect thread_safe_state as a kwarg
                # The lambda in tools.py handles this adaptation usually.
                try:
                    return AVAILABLE_TOOLS[tool_name](**tool_args)
                except Exception as e:
                    return {"status": "error", "message": f"Worker Tool Execution Error for {tool_name}: {e}"}
            else:
                 return {"status": "error", "message": f"Tool {tool_name} not found"}
        
        # Import Collaborator locally
        try:
            from agent import HybridGraphAdvisorCollaborator
        except ImportError:
             import sys
             sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
             from agent import HybridGraphAdvisorCollaborator
             
        collaborator = HybridGraphAdvisorCollaborator(agent, execute_tool_func=worker_tool_executor)
        
        # We need an async loop to run the async run_collaboration method
        # Celery tasks are sync by default, so we wrap it.
        result = asyncio.run(collaborator.run_collaboration(thread_safe_state=thread_safe_state))
        
        log_agent_artifact(run_id, "graph_maintenance_report.md", str(result.get("report", "No report")))
        
        return result
        
    except Exception as e:
        err = f"[Graph-Worker] Failed: {e}"
        log(err)
        return err

@app.task
def run_daily_report_task(run_id: str = "daily_report"):
    """
    Celery Task for Daily Reporting.
    Triggers the generate_scientific_report tool.
    """
    try:
        agent = _rehydrate_agent()
        log(f"[Report-Worker] Starting daily report generation...")
        
        from tools import AVAILABLE_TOOLS
        
        # We need to construct a robust 'thread_safe_state' mock since we are offline/detached.
        from state_manager import load_state
        
        class MockState:
            def __init__(self):
                self.results_log = []
                self.conversation_history = []
                self.strategic_graph_memory = None
                
        state = MockState()
        load_state(state) # Load from disk
        
        state_dict = {
            "results_log": state.results_log,
            "conversation_history": state.conversation_history,
            "strategic_graph_memory": getattr(state, "strategic_graph_memory", None),
            "agent": agent
        }
        
        report_tool = AVAILABLE_TOOLS.get("generate_scientific_report")
        if not report_tool:
            return "Report tool not found."
            
        result = report_tool(
            dataset="combined_sessions", 
            architecture="daily_snapshot",
            thread_safe_state=state_dict
        )
        
        log_agent_artifact(run_id, "daily_report.md", str(result))
        return str(result)
        
    except Exception as e:
        err = f"[Report-Worker] Failed: {e}"
        log(err)
        return err


@app.task
def run_planning_advisor_task(thread_safe_state_snapshot: Dict[str, Any], run_id: str, model_name: str = None):
    """
    Celery Task for the Graph Planner Advisor.
    Runs the Planner logic (Critic -> Planner -> Advice) in the background.
    Returns the 'Advice String' to be consumed by the Decider.
    """
    # Enforce Tracking URI consistency
    from .mlflow_utils import MLFLOW_TRACKING_URI as _canonical_uri, bootstrap_mlflow
    bootstrap_mlflow()
    mlflow.set_tracking_uri(_canonical_uri)

    agent = _rehydrate_agent(model_override=model_name)
    log(f"[Planner-Worker] Starting Advisor Task for Run ID: {run_id}")
    
    # Wait for run to appear
    run_exists = wait_for_run_propagation(run_id)
    
    try:
        # 1. Imports (lazy)
        # We need to add the parent directory to sys.path to ensure modules are found
        import sys
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.append(project_root)

        from context_planner import ContextCriticAgent, StrategicAdvisorTeam
        from graph_memory import StrategicGraphMemory
        from memory_utils import create_vector_memory
        
        # 2. Rehydrate Memories
        # Graph Memory (Load from disk)
        graph_memory = StrategicGraphMemory() # Reads from 'strategic_memory.json'
        
        # Vector Memory (Load from disk/sqlite)
        # We use sqlite backend as it supports concurrency (mostly)
        vector_memory = create_vector_memory(backend="sqlite", fallback_to_json=True)
        
        # 3. Context Critic: Summarize
        critic = ContextCriticAgent(agent)
        
        # We must call async methods. Celery is sync, so we use asyncio.run
        # But we need a wrapper because we have multiple awaits
        
        async def _run_planner_logic():
            # Summarize
            context_summary = await critic.summarize_context(
                thread_safe_state_snapshot,
                graph_memory
            )
            
            # Advice
            advisor = StrategicAdvisorTeam(
                agent,
                graph_memory,
                vector_memory=vector_memory
            )
            
            # Generate Advice using MOSAN logic (Metric Specialists + Pareto)
            # We pass snapshot so it can encode state
            advice_string = await advisor.generate_advice(
                context_summary, 
                thread_safe_state_snapshot
            )
                  
            return advice_string, context_summary

        # Execute Async Logic
        advice, ctx_summary = asyncio.run(_run_planner_logic())
        
        # 4. Logging
        log(f"[Planner-Worker] Advice generated: {advice[:100]}...")
        
        # MLFlow Logging
        if run_exists:
            try:
                with mlflow.start_run(run_id=run_id):
                    with mlflow.start_run(run_name="StrategicAdvisor", nested=True):
                        mlflow.log_text(advice, "output/strategic_advice.md")
                        mlflow.log_text(json.dumps(ctx_summary, indent=2), "debug/context_summary.json")
            except Exception as ml_err:
                log(f"[Planner-Worker] Warning: MLFlow logging failed: {ml_err}")
        
        return advice

    except Exception as e:
        err_msg = f"Graph Planner Advisor failed: {str(e)}"
        log(f"[Planner-Worker] ERROR: {err_msg}")
        return None  # Return None so Decider sees it as "Inactive" logic

