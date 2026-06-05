

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
    Orchestrates pathfinding + expert analysis to provide hybrid strategic advice.
    Combines algorithmic path-finding with LLM expert analysis (~1k tokens total).
    
    Args:
        strategic_intent: Optional intent (e.g., "exploration", "exploitation", "balanced")
        thread_safe_state: Application state
        
    Returns:
        Dict with 'status', 'algorithmic_path', 'expert_analysis', and 'synthesis'
    """
    # This is a simplified version - full implementation will be in HybridGraphAdvisorCollaborator
    # For now, we combine the two tools
    
    path_result = get_strategic_path(thread_safe_state=thread_safe_state)
    transitions_result = get_local_transitions(top_k=5, thread_safe_state=thread_safe_state)
    
    if path_result.get("status") != "completed" or transitions_result.get("status") != "completed":
        return {
            "status": "error",
            "message": "Failed to generate hybrid advice. Check graph state.",
            "path_result": path_result,
            "transitions_result": transitions_result
        }
    
    # Simple synthesis (full version will use LLM)
    path = path_result.get("path", [])
    transitions = transitions_result.get("transitions", [])
    
    if not path and not transitions:
        return {
            "status": "completed",
            "message": "Insufficient graph data for recommendations.",
            "advice": "Continue exploration to build graph knowledge."
        }
    
    # Check if top transition matches first step of path
    agreement = False
    if path and transitions:
        first_step_action = path[0]["action"]
        top_transition_action = transitions[0]["action"]
        agreement = (first_step_action == top_transition_action)
    
    synthesis = {
        "primary_recommendation": path[0] if path else transitions[0] if transitions else None,
        "confidence": "high" if agreement else "medium",
        "rationale": "Both algorithmic path and local Q-values agree" if agreement else "Best local transition based on Q-value",
        "full_path": path[:3] if path else [],  # Show first 3 steps
        "alternatives": transitions[1:3] if len(transitions) > 1 else []
    }
    
    return {
        "status": "completed",
        "synthesis": synthesis,
        "expected_improvement": path_result.get("expected_improvement", 0),
        "strategic_intent": strategic_intent or "balanced"
    }
