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

def _rehydrate_agent(model_override: str = None):
    """
    Creates a new instance of ConversationalAgent using config.yaml.
    This runs inside the Celery Worker process.
    """
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

    # Extract API details based on provider
    provider = cfg.api.get("provider", "Openrouter")
    
    if provider == "Openrouter":
        base_url = "https://openrouter.ai/api/v1"
        api_key = cfg.api.get("openrouter_api_key")
        model_name = cfg.api.get("openrouter_model_name")
    elif provider == "LM Studio (Local)": 
        base_url = "http://host.docker.internal:1234/v1"
        api_key = "not-needed"
        model_name = "local-model"
    elif provider == "vllm":
        base_url = cfg.api.get("local_llm_url", "http://localhost:8000/v1")
        api_key = "EMPTY"
        model_name = "local-model" # Default, overridden below if passed
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
def run_analyst_task(initial_context: str, thread_safe_state: Dict[str, Any], run_id: str, model_name: str = None):
    """
    Celery Task for the Analyst Agent.
    Executes fast analysis tools and summarizes facts.
    """
    agent = _rehydrate_agent(model_override=model_name)
    log(f"[Analyst-Worker] Starting task for Run ID: {run_id} (Model: {agent.model_name})")

    # --- 1. Automated Data Gathering ---
    current_model = thread_safe_state.get("selected_model")
    analysis_tools_requests = [
        {"tool_name": "get_recent_trials", "args": {"n": 5}},
        {"tool_name": "analyze_best_vs_worst_trials", "args": {}},
        {"tool_name": "get_optimization_status", "args": {}},
        {"tool_name": "get_correlation_matrix", "args": {}} 
    ]
    
    gathered_data = []
    
    for tool_req in analysis_tools_requests:
        tool_name = tool_req["tool_name"]
        args = tool_req["args"].copy()
        
        # Smart Filter Injection
        if "architecture_filter" not in args or args["architecture_filter"] is None:
            if current_model:
                args["architecture_filter"] = current_model
        
        try:
            # Execute tool directly using the AVAILABLE_TOOLS registry
            if tool_name in AVAILABLE_TOOLS:
                tool_func = AVAILABLE_TOOLS[tool_name]
                
                try:
                    # Attempt 1: Call with filter
                    result = tool_func(thread_safe_state=thread_safe_state, **args)
                except TypeError as e:
                    if "unexpected keyword argument" in str(e) and "architecture_filter" in args:
                        # Fallback: Retry without architecture_filter (for tools like get_optimization_status)
                        args.pop("architecture_filter")
                        result = tool_func(thread_safe_state=thread_safe_state, **args)
                    else:
                        raise e
                
                filter_used = args.get('architecture_filter', 'Global')
                gathered_data.append(f"--- {tool_name} Results ({filter_used}) ---\n{json.dumps(result, indent=2)}")
            else:
                 gathered_data.append(f"--- {tool_name} Error ---\nTool not found in worker registry.")

        except Exception as e:
            gathered_data.append(f"--- {tool_name} Error ---\n{str(e)}")

    full_context = f"{initial_context}\n\n" + "\n\n".join(gathered_data)

    # --- 2. LLM Summary ---
    analyst_system_prompt = """
    You are part of the Strategy Team, you are a meticulous Data Analyst. Your sole task is to summarize the objective facts based on the provided data context.
     be mindful if the data is about image, text or time series.
     When you get void data it means it is a fresh start, do not PANIC, it is not a faulty system.
     
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
        # Resume parent run solely to nest the child run under it
        with mlflow.start_run(run_id=run_id):
            with mlflow.start_run(run_name="Analyst", nested=True):
                mlflow.log_text(analysis, "output/analyst_report.md")
                mlflow.log_text(full_context, "debug/full_context.txt")
                log(f"[Analyst-Worker] Final Analysis logged to MLFlow (Child Run)")

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
        err_msg = f"Analyst failed: {str(e)}"
        log_agent_artifact(run_id, "analyst_error.txt", err_msg)
        return err_msg

@app.task
def run_strategist_task(conversation_history: List[Dict], run_id: str, model_name: str = None, analyst_report: str = None):
    """
    Celery Task for the Strategist Agent.
    Filters history and generates strategic hypotheses.
    Now accepts 'analyst_report' for sequential intelligence.
    """
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
    strategist_system_prompt = """
    You are the Strategist, **a researcher driven by discovery**, part of the Strategy Team. Your ONLY job is to write a natural language hypothesis.
    As the Strategist, your role is to weave a narrative based on the Analyst's facts, the conversation history, and the available tools. 
    
    **THE GOAL:** Explain the current situation and lead to a clear **Strategic Goal** for the Decider.

    **CRITICAL: PHASES OF STRATEGY**
    1. **PHASE 1: BASELINE ESTABLISHMENT (Priority)**
       - **Fresh Start:** If "0 trials completed" or starting fresh, **DO NOT** suggest Supernet.
       - **Goal:** "Establish a strong baseline using standard architectures (e.g., SimpleCNN, ResNet-like). Target > 90% accuracy."

    2. **PHASE 2: OPTIMIZATION (Fine-Tuning)**
       - If a decent baseline exists (Acc > 85%), focus on Hyperparameter Optimization (Optuna).
       - Target: Push the baseline to its limit.

    3. **PHASE 3: DISCOVERY (Supernet/NAS) - LAST RESORT**
       - **Trigger:** Suggest **Supernet Search** ONLY if:
         a) Accuracy is stuck below target (e.g., < 95%) AND
         b) Standard optimization has plateaued (diminishing returns).
       - **Warning:** Do not jump to Phase 3 prematurely.

    **HIGHEST PRIORITY - NEW ARCHITECTURE DETECTED:**
    - **IF A NEW MODEL IS FOUND:** If the history shows `[PostSupernetAgent Report]` or that a new model file has been constructed, **YOU MUST SWITCH TO EXPLOITATION.**
      - **Goal:** "A new Genotype has been automatically constructed and verified. Goal: Analyze the initial Optuna Sweep results and proceed with targeted optimization (e.g. Fine-Tuning or extended training) for this new architecture."

    **Formulate the Goal (Implicit Guidance):**
    State your Strategic Goal as a clear objective. While you don't call tools directly, your goal should imply the necessary action for the Decider.

    **CRITICAL RULES:**
    1.  **NO Tool Calls:** You **MUST NOT** generate JSON or tool tags (like <tool_code>). Just text.
    2.  **Be Directive:** Tell the Decider *what* to achieve.
    3.  **Context Bridge:** If you see a "Genotype", NEVER suggest running Supernet again. If the model is already built, focus on OPTIMIZATION.
    
    ### NEW INTELLIGENCE BRIEFING (FROM ANALYST):
    {analyst_report}
    
    Now, formulate your hypothesis based on the contents of the 'NEW INTELLIGENCE BRIEFING' and the history.
    """.replace("{analyst_report}", str(analyst_report) if analyst_report else "No fresh intelligence provided.")

    try:
        hypothesis, _ = agent.run_turn(
            strategist_system_prompt,
            filtered_history
        )
        
        # --- 3. MLFlow Logging (Child Run) ---
        with mlflow.start_run(run_id=run_id):
            with mlflow.start_run(run_name="Strategist", nested=True):
                mlflow.log_text(hypothesis, "output/strategist_plan.md")
                mlflow.log_text(conversation_history_str, "debug/history_snapshot.txt")
                log(f"[Strategist-Worker] Final Hypothesis logged to MLFlow (Child Run)")

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
        err_msg = f"Strategist failed: {str(e)}"
        log_agent_artifact(run_id, "strategist_error.txt", err_msg)
        return err_msg

@app.task
def run_strategic_graph_maintenance_task(run_id: str = "periodic"):
    """
    Celery Task for Graph Maintenance.
    Runs the StrategicGraphEvaluatorCollaborator to refine the graph.
    """
    try:
        agent = _rehydrate_agent()
        log(f"[Graph-Worker] Starting maintenance task...")
        
        # --- FIX: Initialize Graph Memory explicitly in Worker ---
        from graph_memory import StrategicGraphMemory
        
        # This loads from disk!
        graph_memory = StrategicGraphMemory()
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
            from agent import StrategicGraphEvaluatorCollaborator
        except ImportError:
             import sys
             sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
             from agent import StrategicGraphEvaluatorCollaborator
             
        collaborator = StrategicGraphEvaluatorCollaborator(agent, execute_tool_func=worker_tool_executor)
        
        # We need an async loop to run the async run_collaboration method
        # Celery tasks are sync by default, so we wrap it.
        result = asyncio.run(collaborator.run_collaboration())
        
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
