
# Context-Aware Graph Planner Classes
# Added: 2025-12-23 for context-aware planning with graph memory

import asyncio
import json
import re
from typing import Dict, Any, List, Optional
import sys

def log(message: str):
    """Simple logger to stderr."""
    print(message, file=sys.stderr)


class ContextCriticAgent:
    """
    Critic agent that summarizes current context and validates plans.
    Acts as a filter between raw state and the planner, ensuring context awareness.
    """
    
    def __init__(self, main_agent: 'ConversationalAgent'):
        self.main_agent = main_agent
    
    async def summarize_context(
        self, 
        thread_safe_state: Dict[str, Any],
        strategic_graph: Any  # StrategicGraphMemory
    ) -> Dict[str, Any]:
        """
        Generate a concise context summary for the planner.
        
        Args:
            thread_safe_state: Current session state
            strategic_graph: StrategicGraphMemory instance
            
        Returns:
            Dict with context summary including constraints, errors, status
        """
        # 1. Extract todo status
        todos = thread_safe_state.get("todos", [])
        total_todos = len(todos)
        completed = len([t for t in todos if isinstance(t, dict) and t.get("status") == "completed"])
        in_progress = len([t for t in todos if isinstance(t, dict) and t.get("status") == "in_progress"])
        todo_status = f"{completed}/{total_todos} completed, {in_progress} in progress"
        
        # 2. Extract recent errors (last 3 from conversation_history)
        recent_errors = self._extract_recent_errors(
            thread_safe_state.get("conversation_history", []),
            n=3
        )
        
        # 4. Get active constraints
        constraints = {
            "max_concurrent_gpu": thread_safe_state.get("max_concurrent_gpu", 1),
            "selected_model": thread_safe_state.get("selected_model", "unknown"),
            "optimization_target": thread_safe_state.get("optimization_target", 0.99),
            "tool_blacklist": thread_safe_state.get("tool_blacklist", {})
        }
        
        # 4b. Get Available Data Inventory (Critical Fix for Hallucinations)
        # We need to know what data actually exists on disk to prevent proposing optimization on non-existent data.
        from tools import list_available_datasets
        try:
            available_datasets = list_available_datasets(thread_safe_state)
            # Flatten to just a list of valid (manifest, representation) pairs or similar simple summary
            data_inventory = []
            if isinstance(available_datasets, dict) and "datasets" in available_datasets:
                raw_datasets = available_datasets["datasets"]
                # `datasets` is a dict {manifest_name: manifest_dict}, NOT a list.
                # Iterating a dict directly yields its keys (strings), causing
                # AttributeError: 'str' object has no attribute 'get'.
                # We must use .items() to get (name, value) pairs.
                if isinstance(raw_datasets, dict):
                    for manifest_name, ds in raw_datasets.items():
                        if not isinstance(ds, dict):
                            # Corrupted entry — include the name with a warning tag
                            data_inventory.append(f"{manifest_name}:(malformed entry)")
                            continue
                        reps = ds.get("generated_representations", {})
                        if reps:
                            for r in reps.keys():
                                data_inventory.append(f"{manifest_name}:{r}")
                        else:
                            data_inventory.append(f"{manifest_name}:(no representations)")
                elif isinstance(raw_datasets, list):
                    # Legacy / unexpected list format — best-effort fallback
                    for ds in raw_datasets:
                        if isinstance(ds, dict):
                            name = ds.get("manifest_name", "unknown")
                            reps = ds.get("generated_representations", {})
                            if isinstance(reps, dict):
                                for r in reps.keys():
                                    data_inventory.append(f"{name}:{r}")
                            elif isinstance(reps, list):
                                for r in reps:
                                    data_inventory.append(f"{name}:{r}")
        except Exception as e:
            data_inventory = [f"Error checking data: {e}"]

        # 5. Get graph performance summary
        graph_summary = self._get_graph_performance(strategic_graph)
        
        # Prepare Prompt Components
        graph_section = f"- Graph Performance: {graph_summary}\n" if graph_summary else ""

        # 6. Call LLM to synthesize
        critic_prompt = f"""You are the **Context Critic** - create a precise, actionable summary of the current state.

**INPUT DATA:**
- Todo Status: {todo_status}
- Recent Errors (last 3):
{json.dumps(recent_errors, indent=2)}

- Active Constraints:
{json.dumps(constraints, indent=2)}

- Data Inventory (Manifest:Representation):
{json.dumps(data_inventory, indent=2)}

{graph_section}
- Current Architecture: {thread_safe_state.get('selected_model', 'unknown')}
- Best Accuracy: {thread_safe_state.get('best_accuracy', 0.0):.4f}
- Global Step: {thread_safe_state.get('global_step_counter', 0)}

**OUTPUT FORMAT (max 150 words, JSON only):**
{{
  "active_constraints": ["constraint1", "constraint2", ...],
  "errors_to_avoid": [
    {{"tool": "tool_name", "context": "when_this_happens", "reason": "why_failed"}},
    ...
  ],
  "available_data": ["manifest:rep", ...],
  "context_prior": "One sentence describing current task and goal",
  "todo_status": "X/Y completed",
  "task_type": "architecture_exploration | optimization | debugging",
  "graph_insights": "One sentence on what's working from graph"
}}

Output ONLY valid JSON, no markdown, no explanation."""

        try:
            summary_text, _ = await asyncio.to_thread(
                self.main_agent.run_turn,
                "You are a precise context analyzer. Output ONLY valid JSON.",
                [{"role": "user", "parts": [critic_prompt]}]
            )
            
            # Parse JSON response
            parsed = self._parse_context_summary(summary_text)
            return parsed
            
        except Exception as e:
            log(f"[Context Critic] Error in summarize_context: {e}")
            # Return safe default
            return {
                "active_constraints": [f"max_gpu={constraints['max_concurrent_gpu']}"],
                "errors_to_avoid": recent_errors[:2],
                "context_prior": f"Working on {constraints['selected_model']} optimization",
                "todo_status": todo_status,
                "task_type": "optimization",
                "graph_insights": "Insufficient data"
            }
    
    async def validate_plan(
        self,
        plan: Dict[str, Any],
        context_summary: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Validate a proposed plan against current context and constraints.
        
        Args:
            plan: The proposed plan from GraphPlannerAgent
            context_summary: The context summary from summarize_context
            
        Returns:
            Dict with decision ("APPROVED"/"MODIFY"/"REJECT") and reason
        """
        validation_prompt = f"""You are the **Plan Validator**. Check if this plan is safe and appropriate.

**PLAN:**
{json.dumps(plan, indent=2)}

**CURRENT CONTEXT:**
{json.dumps(context_summary, indent=2)}

**VALIDATION RULES:**
1. Plan must NOT use tools in errors_to_avoid list
2. Plan must respect active constraints (GPU limits, architecture)
3. Plan must align with current task_type
4. Plan should avoid known error patterns from context
5. **DATA CHECK:** If running 'run_training_trial' or 'run_strategic_optuna_sweep' or 'run_supernet_search', key 'available_data' MUST contain the required manifest:representation. If not, REJECT and suggest 'generate_representation'.

**OUTPUT (JSON only):**
{{
  "decision": "APPROVED" | "MODIFY" | "REJECT",
  "reason": "Brief explanation why",
  "modifications": ["suggestion1", "suggestion2"] or null
}}

Output ONLY valid JSON."""

        try:
            decision_text, _ = await asyncio.to_thread(
                self.main_agent.run_turn,
                "You are a strict plan validator. Output ONLY valid JSON.",
                [{"role": "user", "parts": [validation_prompt]}]
            )
            
            parsed = self._parse_validation_decision(decision_text)
            return parsed
            
        except Exception as e:
            log(f"[Context Critic] Error in validate_plan: {e}")
            # Safe default: approve but warn
            return {
                "decision": "APPROVED",
                "reason": f"Validation failed ({e}), proceeding with caution",
                "modifications": None
            }
    
    def _extract_recent_errors(
        self, 
        conversation_history: List[Dict[str, Any]],
        n: int = 3
    ) -> List[Dict[str, str]]:
        """Extract last N errors from conversation history."""
        errors = []
        
        for msg in reversed(conversation_history):
            if len(errors) >= n:
                break
            
            content = str(msg.get("parts", [""])[0] if msg.get("parts") else "")
            
            # Look for error indicators
            if any(keyword in content.lower() for keyword in ["error", "failed", "exception", "traceback"]):
                # Try to extract tool name
                tool_match = re.search(r'"tool_name":\s*"([^"]+)"', content)
                tool_name = tool_match.group(1) if tool_match else "unknown"
                
                # Extract error reason (simplified)
                error_reason = "execution_failed"
                if "out of memory" in content.lower() or "oom" in content.lower():
                    error_reason = "out_of_memory"
                elif "timeout" in content.lower():
                    error_reason = "timeout"
                elif "invalid" in content.lower():
                    error_reason = "invalid_parameters"
                
                errors.append({
                    "tool": tool_name,
                    "context": "recent_execution",
                    "reason": error_reason
                })
        
        return errors
    
    def _get_graph_performance(self, strategic_graph: Any) -> str:
        """Get a brief summary of graph performance."""
        try:
            if not strategic_graph:
                return None # User requested no trace
            
            if not strategic_graph.graph:
                return "Graph empty (no historical data)"
            
            num_nodes = strategic_graph.graph.number_of_nodes()
            num_edges = strategic_graph.graph.number_of_edges()
            
            # Get avg reward from edges
            rewards = []
            for u, v, key, data in strategic_graph.graph.edges(data=True, keys=True):
                # MOSAN: Reward is index 0 of impact_vector
                if "impact_vector" in data and isinstance(data["impact_vector"], list) and len(data["impact_vector"]) > 0:
                     rewards.append(data["impact_vector"][0])
                elif "avg_reward" in data: # Fallback for legacy
                     rewards.append(data["avg_reward"])
            
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            
            return f"{num_nodes} states, {num_edges} actions, avg_reward={avg_reward:.3f}"
            
        except Exception as e:
            return f"Graph stats unavailable: {e}"
    
    def _parse_context_summary(self, text: str) -> Dict[str, Any]:
        """Parse LLM response into structured context summary."""
        try:
            # Extract JSON from text (may have markdown)
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            log(f"[Context Critic] Failed to parse context summary: {e}")
            # Return minimal valid structure
            return {
                "active_constraints": [],
                "errors_to_avoid": [],
                "context_prior": "Unable to summarize context",
                "todo_status": "unknown",
                "task_type": "unknown",
                "graph_insights": "Parse failed"
            }
    
    def _parse_validation_decision(self, text: str) -> Dict[str, str]:
        """Parse LLM validation response."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                parsed = json.loads(json_match.group(0))
                # Normalize decision
                decision = parsed.get("decision", "APPROVED").upper()
                if decision not in ["APPROVED", "MODIFY", "REJECT"]:
                    decision = "APPROVED"  # Safe default
                
                return {
                    "decision": decision,
                    "reason": parsed.get("reason", "No reason provided"),
                    "modifications": parsed.get("modifications")
                }
        except Exception as e:
            log(f"[Context Critic] Failed to parse validation: {e}")
        
        # Safe default
        return {
            "decision": "APPROVED",
            "reason": "Validation parsing failed, approving by default",
            "modifications": None
        }



from cluster.strategies.strategic_advisor import ParetoRanker, StrategicContext


# Per-lens saturation escalation text templates.
# Appended to advice when ParetoRanker detects the active lens has run out of useful signal.
SATURATION_TEMPLATES = {
    "EXPLORER": (
        "\n\n[SATURATION DETECTED] Graph novelty exhausted "
        "(percentile = {value:.0%}, threshold = {threshold:.0%}).\n"
        "All reachable nodes in the current graph region have been thoroughly explored. "
        "Continuing to rank existing transitions will produce diminishing returns.\n"
        "Recommended escalation: invoke the Innovation Team to author a novel "
        "architecture within the current representational family, generating new graph nodes."
    ),
    "HAWK": (
        "\n\n[SATURATION DETECTED] Accuracy plateau "
        "(recent slope = {value:+.4f}/edge, threshold = {threshold:+.4f}).\n"
        "The last {window} transitions show no meaningful accuracy improvement. "
        "Standard exploitation has run its course on the current architectural family.\n"
        "Recommended escalation: (1) run a focused Optuna sweep on the current best node; "
        "(2) if the sweep also plateaus, switch representational family."
    ),
    "SAGE": (
        "\n\n[SATURATION DETECTED] Stability signal collapse "
        "(recent overfitting std = {value:.4f}, threshold = {threshold:.4f}).\n"
        "Recent transitions produce nearly identical overfitting deltas. "
        "The stability landscape is no longer informative under standard validation.\n"
        "Recommended escalation: run k-fold cross-validation on the top-stability node "
        "to obtain a high-confidence robustness estimate before declaring convergence."
    ),
    "ENGINEER": (
        "\n\n[SATURATION DETECTED] Efficiency coverage gap "
        "(coverage = {value:.0%}, threshold = {threshold:.0%}).\n"
        "A significant fraction of graph edges lack measured latency/memory data. "
        "Efficiency-driven decisions on the current graph are based on incomplete profiling.\n"
        "Recommended escalation: dispatch a profiling pass over non-profiled transitions "
        "before continuing optimization."
    ),
}


def _format_saturation_block(saturation_info: Dict[str, Any], window: int) -> str:
    """Render the lens-specific escalation block, or empty string if not applicable."""
    if not saturation_info or not saturation_info.get("is_saturated"):
        return ""
    lens = saturation_info.get("lens", "")
    template = SATURATION_TEMPLATES.get(lens)
    if not template:
        return ""
    try:
        return template.format(
            value=saturation_info.get("metric_value", 0.0),
            threshold=saturation_info.get("threshold", 0.0),
            window=window,
        )
    except Exception:
        return ""


class StrategicAdvisorTeam:
    """
    MOSAN ADVISOR TEAM 🧠
    Replaces the old GraphPlanner.
    Uses Multi-Objective Reinforcement Learning (Accuracy, Stability, Novelty, Efficiency)
    to advise the Decider on the best next step.
    """
    
    def __init__(
        self,
        main_agent: 'ConversationalAgent',
        strategic_graph: Any,  # StrategicGraphMemory
        vector_memory: Any = None,
        saturation_config: Optional[Dict[str, Any]] = None,
    ):
        self.main_agent = main_agent
        self.graph = strategic_graph
        self.ranker = ParetoRanker()
        # Optional per-lens saturation config; if None, saturation detection is disabled.
        self.saturation_config = saturation_config
        
    async def generate_advice(
        self,
        context_summary: Dict[str, Any],
        thread_safe_state: Dict[str, Any],
        strategic_intent: Optional[str] = None, # Added explicit intent
        default_lens: str = "auto" # Configuration override
    ) -> str:
        """
        Consults the Strategic Graph and generates advice for the Decider.
        """
        try:
            # 1. Determine Contextual Lens
            if strategic_intent:
                lens = self._get_lens_from_intent(strategic_intent)
                lens_name = strategic_intent.upper()
            elif default_lens and default_lens.lower() != "auto":
                # Config has a specific partiality forced
                lens = self._get_lens_from_intent(default_lens)
                lens_name = f"{default_lens.upper()} (Forced)"
            else:
                lens = self._determine_lens(context_summary, thread_safe_state)
                lens_name = self._get_lens_name(lens)
            
            # 2. Encode Current State Vector
            # We need the vector to query the organic graph.
            # Using rl_utils (lazy import to avoid circular dep if any)
            from rl_utils import encode_state
            
            # We need to reconstruct the state vector from thread_safe_state
            # CAUTION: encode_state usually takes the whole state dict + history
            # Let's assume thread_safe_state has enough.
            current_arch = thread_safe_state.get("selected_model", "unknown")
            state_vector = encode_state(thread_safe_state)
            
            # 3. Query Graph for Transitions from Current Anchor
            transitions = self.graph.get_available_transitions(state_vector, current_arch)
            
            if not transitions:
                return (
                    f"**STRATEGIC ADVISOR ({lens_name} Lens):**\n"
                    f"No historical path found from this state for '{current_arch}'.\n"
                    f"Recommendation: **EXPLORE**. Try a new tool or architecture modification."
                )
            
            # 4. Rank Actions using Pareto Logic
            # We need target node data for full bonus calculation (e.g. novelty)
            # Efficiently fetch target node data
            target_nodes_data = []
            for t in transitions:
                node_data = self.graph.get_node_data(t["target_node"])
                target_nodes_data.append(node_data)
                
            ranked_actions, saturation_info = self.ranker.rank_actions(
                transitions, lens, target_nodes_data,
                graph=self.graph,
                saturation_config=self.saturation_config,
            )

            # 5. Format Advice
            advice = f"**STRATEGIC ADVISOR ({lens_name} Lens):**\n"
            advice += "Based on historical Multi-Objective Analysis:\n"
            
            # Show top 3
            for i, (action, score, breakdown) in enumerate(ranked_actions[:3]):
                tool_key = action['action']
                full_args = action.get('full_arguments', {})
                
                display_name = tool_key
                if full_args:
                     # Reconstruct full call signature for the Agent
                     args_str = ", ".join([f"{k}={repr(v)}" for k, v in full_args.items()])
                     base_name = tool_key.split('(')[0]
                     display_name = f"{base_name}({args_str})"
                
                if i == 0:
                     # For the top recommendation, predict future trajectory
                     trajectory = self._predict_trajectory(action['target_node'], lens, depth=2)
                     if trajectory:
                         traj_str = " -> ".join([t['action'] for t in trajectory])
                         advice += f"   - Projected Path: **{display_name}** -> {traj_str}\n"

                advice += f"{i+1}. **{display_name}** (Score: {score:.2f})\n"
                advice += f"   - Impact: Acc={breakdown['acc']:.2f}, Stab={breakdown['stab']:.2f}, Nov={breakdown['nov']:.2f}, Eff={breakdown['eff']:.2f}\n"

            # 6. Per-lens saturation escalation (advisory; no automatic action).
            try:
                window = int((self.saturation_config or {}).get("saturation_recent_window", 10))
                escalation = _format_saturation_block(saturation_info, window)
                if escalation:
                    advice += escalation
                    # Telemetry
                    try:
                        from saturation_logger import log_saturation_event
                        log_saturation_event({
                            "timestamp": __import__("time").time(),
                            "dataset_id": getattr(self.graph, "dataset_id", "unknown"),
                            "lens": saturation_info.get("lens", ""),
                            "metric_name": saturation_info.get("metric_name", ""),
                            "metric_value": saturation_info.get("metric_value", 0.0),
                            "threshold": saturation_info.get("threshold", 0.0),
                            "current_arch": current_arch,
                            "n_graph_nodes": saturation_info.get("n_graph_nodes", 0),
                        })
                    except Exception as _logerr:
                        log(f"[Strategic Advisor] Saturation telemetry skipped: {_logerr}")
            except Exception as _e:
                log(f"[Strategic Advisor] Saturation block rendering skipped: {_e}")

            return advice

        except Exception as e:
            log(f"[Strategic Advisor] Error generating advice: {e}")
            return "**STRATEGIC ADVISOR:** (Offline due to internal error)"

    def _predict_trajectory(self, start_node: str, lens: StrategicContext, depth: int = 2) -> List[Dict]:
        """
        Simulates the optimal path from start_node for 'depth' steps using the given lens.
        """
        trajectory = []
        current_node = start_node
        
        for _ in range(depth):
            # Get transitions directly from the node (we are already at the Anchor)
            if not self.graph.graph.has_node(current_node):
                break
                
            transitions = []
            for u, v, key, data in self.graph.graph.edges(current_node, data=True, keys=True):
                 transitions.append({
                    "action": key,
                    "target_node": v,
                    "impact_vector": data.get("impact_vector", []),
                    "visits": data.get("visits", 0),
                    "raw_rewards": data.get("raw_rewards", []),
                    "full_arguments": data.get("full_arguments", {}),
                    "q_vector": data.get("q_vector", None)
                })
            
            if not transitions:
                break
            
            # Rank
            target_nodes_data = [self.graph.get_node_data(t["target_node"]) for t in transitions]
            ranked, _ = self.ranker.rank_actions(transitions, lens, target_nodes_data)

            if not ranked:
                break

            best_action, score, _ = ranked[0]
            trajectory.append(best_action)
            current_node = best_action["target_node"]
            
        return trajectory

    def _get_lens_from_intent(self, intent: str) -> StrategicContext:
        """Map string intent to Lens."""
        intent = intent.upper()
        if "STAB" in intent: return StrategicContext.get_sage_lens()
        if "NOV" in intent: return StrategicContext.get_explorer_lens()
        if "EFF" in intent: return StrategicContext.get_engineer_lens()
        return StrategicContext.get_hawk_lens() # Default to Accuracy

    def _determine_lens(
        self,
        context_summary: Dict[str, Any],
        thread_safe_state: Optional[Dict[str, Any]] = None,
    ) -> StrategicContext:
        """Data-driven lens selection from results_log signals.

        Priority (highest wins):
          1. High volatility in recent runs      → SAGE  (stabilise first)
          2. Best accuracy already ≥ 0.96        → ENGINEER  (squeeze inference)
          3. Current arch saturated + low arch diversity → EXPLORER
          4. LLM task_type == "debugging"        → SAGE
          5. LLM task_type == "architecture_exploration" → EXPLORER
          6. Default                              → HAWK
        """
        import statistics

        state = thread_safe_state or {}
        results_log = state.get("results_log", []) or []
        selected_model = state.get("selected_model", "")
        best_accuracy = state.get("best_accuracy") or 0.0

        # Normal trials for current arch (exclude finetune — mean_accuracy is None there).
        # Guard: results_log entries must be dicts — skip any corrupted string entries
        # to prevent AttributeError: 'str' object has no attribute 'get'.
        arch_trials = [
            t for t in results_log
            if isinstance(t, dict)
            and t.get("architecture") == selected_model
            and t.get("source") != "finetune_trial"
            and (t.get("mean_accuracy") is not None)
        ]

        # ── Signal 1: Volatility ────────────────────────────────────────────
        # High std in recent 5 trials → training is unstable → SAGE
        VOLATILITY_N = 5
        volatile = False
        if len(arch_trials) >= VOLATILITY_N:
            recent_accs = [(t.get("mean_accuracy") or 0.0) for t in arch_trials[-VOLATILITY_N:]]
            try:
                if statistics.stdev(recent_accs) > 0.03:
                    volatile = True
            except statistics.StatisticsError:
                pass

        # ── Signal 2: High accuracy achieved ────────────────────────────────
        # Already near ceiling → shift focus to inference efficiency
        HIGH_ACC_THRESHOLD = 0.96
        high_accuracy = best_accuracy >= HIGH_ACC_THRESHOLD

        # ── Signal 3: Saturation + low arch diversity ────────────────────────
        # Current arch has many trials with no recent improvement AND we haven't
        # explored many distinct architectures → EXPLORER to escape local optima
        SATURATION_N = 8
        saturated = False
        if len(arch_trials) >= SATURATION_N:
            recent = arch_trials[-SATURATION_N:]
            recent_best = max((t.get("mean_accuracy") or 0.0) for t in recent)
            global_best = max((t.get("mean_accuracy") or 0.0) for t in arch_trials)
            if global_best > 0 and (global_best - recent_best) < 0.002:
                saturated = True

        distinct_archs_tried = len({
            t.get("architecture") for t in results_log
            if t.get("source") != "finetune_trial"
            and t.get("mean_accuracy") is not None
            and t.get("architecture")
        })
        low_diversity = distinct_archs_tried < 4

        # ── Decision tree ────────────────────────────────────────────────────
        if volatile:
            return StrategicContext.get_sage_lens()

        if high_accuracy:
            return StrategicContext.get_engineer_lens()

        if saturated and low_diversity:
            return StrategicContext.get_explorer_lens()

        # Fall back to LLM task_type as secondary signal
        task_type = context_summary.get("task_type", "optimization")
        if task_type == "debugging":
            return StrategicContext.get_sage_lens()
        if task_type == "architecture_exploration":
            return StrategicContext.get_explorer_lens()

        return StrategicContext.get_hawk_lens()

    def _get_lens_name(self, lens: StrategicContext) -> str:
        if lens.weights['acc'] > 0.8: return "HAWK"
        if lens.weights['stab'] > 0.5: return "SAGE"
        if lens.weights['nov'] > 0.5: return "EXPLORER"
        if lens.weights['eff'] > 0.4: return "ENGINEER"
        return "CUSTOM"
