import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

# Use relative import for robustness in different running contexts
try:
    from state_manager import PERSISTENT_PATHS
    from ui_logger import log
except ImportError:
    # Fallback for worker context if path not setup
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from state_manager import PERSISTENT_PATHS
    from ui_logger import log

def save_agent_trace(
    run_id: str,
    agent_name: str,
    system_prompt: str,
    inputs: Any,
    output: Any,
    thoughts: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Saves a detailed execution trace of an agent turn to a JSON file.
    
    Args:
        run_id: Unique identifier for the run/step.
        agent_name: Name of the agent (e.g., "Analyst", "Strategist").
        system_prompt: The system prompt used.
        inputs: The input data or messages passed to the agent.
        output: The final output/response.
        thoughts: Optional internal thoughts or Chain-of-Thought.
        metadata: Additional context (e.g., model name, duration).
        
    Returns:
        The path to the saved file.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure directory exists (redundant check for safety)
        trace_dir = PERSISTENT_PATHS.get("detailed_traces_dir")
        if not trace_dir:
            # Fallback
            trace_dir = os.path.join(os.path.dirname(__file__), "..", "processed_data", "logs", "detailed_traces")
        
        os.makedirs(trace_dir, exist_ok=True)
        
        filename = f"{timestamp}_{agent_name}_{run_id}.json"
        filepath = os.path.join(trace_dir, filename)
        
        trace_data = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "agent_name": agent_name,
            "system_prompt": system_prompt,
            "inputs": inputs,
            "output": output,
            "thoughts": thoughts,
            "metadata": metadata or {}
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
            
        log(f"[Trace Logger] Saved detailed trace for {agent_name} to {filename}")
        return filepath
        
    except Exception as e:
        log(f"[Trace Logger] ERROR: Failed to save trace: {e}")
        return ""
