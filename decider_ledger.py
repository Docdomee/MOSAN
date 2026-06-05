# decider_ledger.py
"""
Single Ledger Architecture — Phase 3: Ledger Decider.

The Decider reads Part 3 (Strategic Plan & Objectives) from the ledger,
executes tools in a budget-bounded loop, and writes Part 4 (Execution Log).

Contract: The Decider only EXECUTES. No planning, no research, no theorizing.
It reads objectives from the ledger and calls tools to fulfill them.
"""

import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ledger_schema import (
    append_section,
    append_raw,
    parse_section,
    read_ledger,
    SECTION_MARKERS,
)
from ui_logger import log
from cluster.prompts import LEDGER_DECIDER_SYSTEM_PROMPT
import rl_utils


# ---------------------------------------------------------------------------
# Metric alias resolution
# ---------------------------------------------------------------------------

_METRIC_ALIASES: Dict[str, str] = {
    "val_accuracy": "mean_accuracy",
    "test_accuracy": "mean_accuracy",
    "accuracy": "mean_accuracy",
    "val_acc": "mean_accuracy",
    "best_val_accuracy": "best_accuracy",
}


def _resolve_accuracy(result: dict) -> Optional[float]:
    """Return the first accuracy value found, checking canonical keys, aliases, then any key containing 'accuracy'."""
    for key in ("mean_accuracy", "best_accuracy"):
        v = result.get(key)
        if v is not None:
            return v
    for alias in _METRIC_ALIASES:
        v = result.get(alias)
        if v is not None:
            return v
    for key, v in result.items():
        if "accuracy" in key and v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_budget(part3_content: str) -> int:
    """Extract step budget from Part 3 content."""
    match = re.search(r"\*\*Step Budget:\*\*\s*(\d+)", part3_content)
    if match:
        return int(match.group(1))
    # Fallback: conservative default
    log("[Decider] WARNING: No step budget found in Part 3. Defaulting to 5.")
    return 5


def _extract_objectives(part3_content: str) -> List[str]:
    """Extract objective lines from Part 3 content."""
    # Match both checked and unchecked: - [ ] O1: ... or - O1: ...
    objectives = re.findall(r"-\s*(?:\[[ x]\]\s*)?O\d+:\s*(.+)", part3_content)
    if objectives:
        return [f"O{i+1}: {obj.strip()}" for i, obj in enumerate(objectives)]

    # Fallback: any bullet points
    bullets = re.findall(r"^[-*]\s+(.+)", part3_content, re.MULTILINE)
    if bullets:
        return [f"O{i+1}: {b.strip()}" for i, b in enumerate(bullets)]

    return []


def _format_execution_history(execution_log: List[Dict]) -> str:
    """Format the execution log for injection into the Decider prompt.
    Failures are shown with full error text so the Decider can reason about them.
    """
    if not execution_log:
        return "(No actions taken yet)"

    lines = []
    for entry in execution_log:
        step = entry["step"]
        tool = entry["tool"]
        success = "OK" if entry["success"] else "FAILED"
        summary = entry.get("result_summary", "No summary")
        args = entry.get("args", {})
        # Always show key args inline for training tools so Decider avoids exact repetition
        key_args = {}
        for k in ("representation_type", "custom_architecture_file", "manifest_name",
                   "num_conv_layers", "filters", "learning_rate", "dropout_rate"):
            if k in args:
                key_args[k] = args[k]
        args_hint = f" | args={json.dumps(key_args)}" if key_args else ""
        lines.append(f"  Step {step}: {tool} → [{success}]{args_hint}")
        lines.append(f"           Result: {summary}")
        if not entry["success"]:
            lines.append(f"           ⚠️ FULL ERROR: {summary}")

    return "\n".join(lines)


def _summarize_result(result: Any) -> str:
    """Create a summary of a tool result for the execution log.
    Error messages are preserved in full so the next cycle can diagnose failures.
    """
    if isinstance(result, dict):
        status = result.get("status", "unknown")
        message = result.get("message", "")
        acc = _resolve_accuracy(result)
        if acc is not None:
            # Include held_out separately from CV so they are never confused
            ho = result.get("held_out_test_accuracy")
            cv = result.get("mean_accuracy")
            parts = [f"{status}"]
            if ho is not None:
                parts.append(f"HO={ho:.4f}")
            if cv is not None:
                parts.append(f"CV={cv:.4f}")
            if ho is None and cv is None:
                parts.append(f"accuracy={acc}")
            if message:
                parts.append(f"msg={str(message)[:200]}")
            return " | ".join(parts)
        if message:
            # Preserve full error message — never truncate error text
            return f"{status}: {str(message)}"
        return f"{status}"
    # Non-dict result: preserve fully (errors must not be truncated)
    return str(result)


# ---------------------------------------------------------------------------
# Main Decider loop
# ---------------------------------------------------------------------------

def run_decider(
    ledger_path: str,
    thread_safe_state: Dict[str, Any],
    execute_tool_func: Optional[Callable] = None,
    constraints: Optional[List[str]] = None,
    prev_ledger_path: Optional[str] = None,
) -> bool:
    """
    Run the Ledger Decider: read Part 3, execute tools, write Part 4.

    The Decider operates in a budget-bounded loop:
    1. Build prompt from objectives + execution history
    2. LLM picks tool(s) to call
    3. Execute tools via execute_tool_func
    4. Log results, decrement budget
    5. Repeat until budget exhausted or LLM signals "done"

    Args:
        ledger_path: Path to the current cycle's ledger (Parts 1-3 must exist).
        thread_safe_state: The shared state proxy.
        execute_tool_func: Tool dispatcher (main.py's execute_tool).
        constraints: User directive constraints to inject into prompt.

    Returns:
        True if execution completed (budget spent or objectives met), False on error.
    """
    log("[Decider] Starting Ledger Decider execution...")

    # --- Checkpoint: skip if Part 4 already exists ---
    ledger_content = read_ledger(ledger_path)
    if SECTION_MARKERS["part_4"] in ledger_content:
        log("[Decider] Part 4 already exists — skipping (checkpoint recovery).")
        return True

    # --- Extract Part 3 ---
    # Find Part 3 content (everything after ## Part 3 header)
    part3_match = re.search(
        r"## Part 3: Strategic Plan & Objectives\s*\n(.*)",
        ledger_content,
        re.DOTALL,
    )
    if not part3_match:
        log("[Decider] ERROR: Part 3 not found in ledger. Cannot proceed.")
        return False

    part3_content = part3_match.group(1).strip()
    total_budget = _extract_budget(part3_content)
    objectives = _extract_objectives(part3_content)

    if not objectives:
        # Fallback: search globally for **Semantic Objectives:** block (since Part 3 is now lean)
        obj_match = re.search(
            r"\*\*Semantic Objectives:\*\*\s*\n(.*?)(?=\n##|\n\*\*|\Z)",
            ledger_content,
            re.DOTALL,
        )
        if obj_match:
            objectives = _extract_objectives(obj_match.group(1).strip())

    if not objectives:
        log("[Decider] ERROR: No objectives found in Part 3 or globally in Part 2.")
        _write_part4(ledger_path, [], objectives, total_budget, "No objectives found")
        return True

    log(f"[Decider] Budget: {total_budget} steps | Objectives: {len(objectives)}")
    for obj in objectives:
        log(f"[Decider]   - {obj}")

    # --- Get agent and tool dispatcher ---
    # Use dedicated decider_agent if configured (fast non-thinking model),
    # otherwise fall back to the main agent.
    agent = thread_safe_state.get("decider_agent") or thread_safe_state.get("agent")
    if agent is None:
        log("[Decider] ERROR: No agent found in state.")
        return False

    if execute_tool_func is None:
        log("[Decider] ERROR: No execute_tool_func provided.")
        return False

    # --- Discover architecture library templates ---
    library_listing = "None available."
    try:
        import os
        library_path = os.path.join(
            os.path.abspath(os.path.join(os.path.dirname(__file__), "cluster", "library", "architectures"))
        )
        if os.path.isdir(library_path):
            arch_files = [f.replace(".py", "") for f in os.listdir(library_path) if f.endswith(".py")]
            if arch_files:
                library_listing = ", ".join(sorted(arch_files))
                log(f"[Decider] Architecture library: {library_listing}")
    except Exception as e:
        log(f"[Decider] Warning: Could not read architecture library: {e}")

    # --- Gather cross-cycle context ---
    current_model = thread_safe_state.get("selected_model", "unknown")
    metadata_info = (
        f"{current_model} on "
        f"{thread_safe_state.get('raw_data_source', 'unknown dataset')}"
    )

    # NOTE: the Research Chronicle (Part 0) is intentionally NOT read here. It is consumed
    # by the Theorist (planner) — see theorist._run_hypothesis_generation. Injecting it into
    # the executor's conversation only diluted context without helping the Decider run the plan.

    recent_cycle_history = "(No previous cycle — this is the first run.)"
    critic_warning = "(No critic warning available — first cycle or summarizer not yet run.)"
    if prev_ledger_path:
        try:
            prev_part4 = parse_section(prev_ledger_path, "Part 4: Decider's Execution Log")
            if prev_part4:
                # Truncate to avoid token bloat — keep last 60 lines
                lines = prev_part4.strip().splitlines()
                if len(lines) > 60:
                    lines = ["...(truncated)..."] + lines[-60:]
                recent_cycle_history = "\n".join(lines)
            else:
                recent_cycle_history = "(Previous cycle ledger exists but Part 4 not found yet.)"
        except Exception as e:
            log(f"[Decider] Warning: Could not read prev cycle history: {e}")

    # --- Load Critic warning from last appendix ---
    try:
        import re as _re
        from ledger_schema import WORKSPACE_DIR as _WS_DIR
        _appendix_files = sorted(
            [f for f in os.listdir(_WS_DIR) if _re.match(r"ledger_appendix_\d+\.md", f)],
            key=lambda x: int(_re.search(r"(\d+)", x).group(1))
        )
        if _appendix_files:
            _last_appendix = os.path.join(_WS_DIR, _appendix_files[-1])
            with open(_last_appendix, "r", encoding="utf-8") as _f:
                _appendix_content = _f.read()
            # Extract critic section if present, else use full content truncated
            _critic_match = _re.search(r"\*\*Critique:\*\*(.*?)(?=\n#|\Z)", _appendix_content, _re.DOTALL)
            if _critic_match:
                critic_warning = _critic_match.group(1).strip()[:600]
            else:
                critic_warning = _appendix_content.strip()[:600]
            log(f"[Decider] Critic warning loaded from {_appendix_files[-1]}.")
    except Exception as e:
        log(f"[Decider] Warning: Could not load critic warning from appendix: {e}")

    # --- Budget-bounded execution loop ---
    execution_log: List[Dict] = []
    steps_remaining = total_budget
    constraints_text = "\n".join(f"- {c}" for c in constraints) if constraints else "(None)"
    objectives_text = "\n".join(f"- {obj}" for obj in objectives)
    halt_reason = None

    # Decider conversation starts clean — the Research Chronicle is consumed by the
    # Theorist (planner), not the executor, to avoid context dilution.
    conversation_history = []

    # Open Part 4 NOW (before any tool runs) and append each step as it completes, so an
    # interrupt mid-cycle still leaves every executed action on disk (append-only).
    _write_part4_header(ledger_path, total_budget)

    while steps_remaining > 0:
        execution_log_batch: List[Dict] = []
        # Refresh current_model from state — may have been updated by auto-switch mid-cycle.
        current_model = thread_safe_state.get("selected_model", current_model)

        # Build innovation context — re-read each iteration so it reflects any switch
        # that happened during this cycle (e.g. propose_intelligent_architecture just ran).
        _last_innovation = thread_safe_state.get("last_innovation_result")
        if _last_innovation and isinstance(_last_innovation, dict) and _last_innovation.get("status") == "success":
            innovation_context = (
                f"The Innovation Team successfully built a new architecture THIS cycle.\n"
                f"- File: `{_last_innovation.get('architecture_file', 'unknown')}`\n"
                f"- Representation: `{_last_innovation.get('representation_type', 'unknown')}`\n"
                f"- Manifest: `{_last_innovation.get('manifest_name', 'unknown')}`\n"
                f"You MUST run `run_training_trial` with `custom_architecture_file` set to the file above "
                f"before calling `<done/>`."
            )
        else:
            innovation_context = "(no new architecture built this cycle)"

        # Build/append prompt to conversation history
        if not execution_log:
            # First turn: inject the full prompt with initial state
            prompt = LEDGER_DECIDER_SYSTEM_PROMPT.format(
                objectives_text=objectives_text,
                steps_remaining=steps_remaining,
                total_budget=total_budget,
                constraints_text=constraints_text,
                execution_history=_format_execution_history(execution_log),
                library_listing=library_listing,
                current_model=current_model,
                recent_cycle_history=recent_cycle_history,
                metadata_info=metadata_info,
                innovation_context=innovation_context,
                critic_warning=critic_warning,
            )
            conversation_history.append({"role": "user", "parts": [prompt]})
        else:
            # Subsequent turns: provide a concise state update
            update_prompt = (
                f"System Update:\n"
                f"- Steps remaining: {steps_remaining} (out of {total_budget})\n"
                f"- Current Model Focus: {current_model}\n"
                f"- Innovation context: {innovation_context}\n"
                f"Please review the previous tool results and continue executing the next objective(s)."
            )
            conversation_history.append({"role": "user", "parts": [update_prompt]})

        try:
            response_text, tool_call_str = agent.run_turn(
                system_prompt="You are the Decider. Execute the plan. Output tool calls or <done/>.",
                conversation_history=conversation_history,
            )
        except Exception as e:
            log(f"[Decider] LLM call failed: {e}")
            halt_reason = f"LLM call failed: {e}"
            break

        if not response_text:
            log("[Decider] LLM returned empty response. Halting.")
            halt_reason = "LLM returned empty response"
            break

        # Append the assistant response to the conversation history so it's not lost
        conversation_history.append({"role": "assistant", "parts": [response_text]})

        # Check for "done" signal — but ONLY if there are no tool calls.
        _has_done_signal = "<done/>" in response_text or (not tool_call_str and "done" in response_text.lower())
        if _has_done_signal and not tool_call_str:
            log("[Decider] LLM signaled completion.")
            break

        # Parse tool calls
        if not tool_call_str:
            log("[Decider] No tool calls in response. Halting.")
            halt_reason = "LLM produced no tool calls"
            break

        tool_calls = agent.parse_tool_calls(tool_call_str)
        if not tool_calls:
            log("[Decider] Failed to parse tool calls. Halting.")
            halt_reason = "Failed to parse tool calls"
            break

        # Hard deduplication: tools that must run at most ONCE per cycle.
        _ONCE_PER_CYCLE_TOOLS = {"list_available_datasets"}
        _tools_called_this_cycle = {e["tool"] for e in execution_log}
        deduplicated_calls = []
        for _call in tool_calls:
            _tname = _call.get("tool_name", _call.get("tool", _call.get("name", "")))
            if _tname in _ONCE_PER_CYCLE_TOOLS and _tname in _tools_called_this_cycle:
                log(f"[Decider] DEDUP: '{_tname}' already called this cycle — skipping duplicate call.")
            else:
                deduplicated_calls.append(_call)
        if len(deduplicated_calls) < len(tool_calls):
            log(f"[Decider] DEDUP: {len(tool_calls) - len(deduplicated_calls)} duplicate call(s) removed.")
        tool_calls = deduplicated_calls
        if not tool_calls:
            log("[Decider] All calls were duplicates — forcing <done/> to end loop.")
            break

        # Split calls into GPU tools (parallelisable) and CPU tools (sequential)
        _GPU_TOOLS = {"run_training_trial", "run_strategic_optuna_sweep", "run_supernet_search"}
        _TRAINING_TOOLS = {"run_training_trial", "run_strategic_optuna_sweep", "run_supernet_search"}

        # --- ARCHITECTURE SWITCH GUARD ---
        _REP_TO_MODEL = {
            "1D_CNN": "1D_CNN",
            "2D_GAF": "2D_GAF",
            "2D_SPECTROGRAM": "2D_SPECTROGRAM",
            "2D_CWT_SCALOGRAM": "2D_CWT_SCALOGRAM",
            "3D_VIDEO": "3D_VIDEO",
            "3D_GAF_VIDEO": "3D_GAF_VIDEO",
            "3D_DYNAMIC_GAF": "3D_DYNAMIC_GAF",
        }
        _already_switching = any(
            c.get("tool_name", c.get("tool", c.get("name", ""))) == "set_optimization_model"
            for c in tool_calls
        )
        if not _already_switching:
            for call in tool_calls:
                t_name = call.get("tool_name", call.get("tool", call.get("name", "")))
                if t_name in _TRAINING_TOOLS:
                    call_args = call.get("args", {})
                    if call_args.get("custom_architecture_file"):
                        break
                    rep_type = call_args.get("representation_type", "")
                    target_model = _REP_TO_MODEL.get(rep_type, "")
                    current_model_now = thread_safe_state.get("selected_model", "")
                    if target_model and target_model != current_model_now:
                        log(f"[Decider] AUTO-GUARD: '{t_name}' requests rep='{rep_type}' but current model is '{current_model_now}'. Injecting set_optimization_model('{target_model}') first.")
                        tool_calls = [{"tool_name": "set_optimization_model", "args": {"model_name": target_model}}] + tool_calls
                        break

        gpu_calls = [c for c in tool_calls if c.get("tool_name", c.get("tool", c.get("name", ""))) in _GPU_TOOLS]
        cpu_calls = [c for c in tool_calls if c not in gpu_calls]

        if steps_remaining <= 0:
            log("[Decider] Budget exhausted.")
            break

        # --- Execute CPU tools sequentially ---
        for call in cpu_calls:
            if steps_remaining <= 0:
                log("[Decider] Budget exhausted mid-batch.")
                break

            tool_name = call.get("tool_name", call.get("tool", call.get("name", "unknown")))
            tool_args = call.get("args", {})
            log(f"[Decider] Step {len(execution_log) + 1}/{total_budget}: {tool_name}")
            try:
                state_before = rl_utils.encode_state(thread_safe_state)
            except Exception:
                state_before = None
            try:
                result = execute_tool_func(tool_name, tool_args, thread_safe_state, None, execute_tool_func)
                success = True
                summary = _summarize_result(result)
            except Exception as e:
                import traceback as _tb
                full_err = _tb.format_exc()
                log(f"[Decider] Tool '{tool_name}' failed: {e}\n{full_err}")
                success = False
                summary = f"Error: {str(e)} | Traceback: {full_err}"
                result = {"status": "error", "message": str(e)}
            try:
                state_after = rl_utils.encode_state(thread_safe_state)
                rl_utils.update_graph_memory(thread_safe_state, tool_name, result, state_before, state_after, tool_args=tool_args)
            except Exception as e:
                log(f"[Decider] Graph update failed for '{tool_name}': {e}")

            _entry = {
                "step": len(execution_log) + 1,
                "tool": tool_name,
                "args": {k: v for k, v in tool_args.items() if k != "thread_safe_state"},
                "success": success,
                "result_summary": summary,
            }
            execution_log.append(_entry)
            _append_part4_step(ledger_path, _entry)  # crash-safe: persist immediately
            execution_log_batch.append({
                "step": len(execution_log),
                "tool": tool_name,
                "success": success,
                "result_raw": json.dumps(result, default=str),
            })
            steps_remaining -= 1

        # --- Execute GPU tools in parallel (one thread per GPU call) ---
        if gpu_calls and steps_remaining > 0:
            max_concurrent = thread_safe_state.get("max_concurrent_gpu", 1)
            batch = gpu_calls[:min(steps_remaining, max_concurrent)]
            if len(gpu_calls) > max_concurrent:
                log(f"[Decider] Capping GPU parallelism to {max_concurrent} (max_concurrent_gpu). {len(gpu_calls) - max_concurrent} call(s) deferred to next iteration.")
            log(f"[Decider] Launching {len(batch)} GPU tool(s) in parallel...")

            import threading as _threading
            _gpu_assign_lock = _threading.Lock()

            def _run_gpu_call(call):
                t_name = call.get("tool_name", call.get("tool", call.get("name", "unknown")))
                t_args = call.get("args", {})
                gpu_id = None

                try:
                    try:
                        _state_before = rl_utils.encode_state(thread_safe_state)
                    except Exception:
                        _state_before = None
                    res = execute_tool_func(t_name, t_args, thread_safe_state, gpu_id, execute_tool_func)
                    try:
                        _state_after = rl_utils.encode_state(thread_safe_state)
                        rl_utils.update_graph_memory(thread_safe_state, t_name, res, _state_before, _state_after, tool_args=t_args)
                    except Exception as ge:
                        log(f"[Decider] Graph update failed for '{t_name}': {ge}")
                    return t_name, t_args, True, _summarize_result(res), res
                except Exception as e:
                    import traceback as _tb
                    full_err = _tb.format_exc()
                    log(f"[Decider] GPU tool '{t_name}' failed: {e}\n{full_err}")
                    return t_name, t_args, False, f"Error: {str(e)} | Traceback: {full_err}", {"status": "error", "message": str(e), "traceback": full_err}
                finally:
                    if gpu_id is not None:
                        try:
                            release_gpu_lock(gpu_id)
                        except Exception:
                            pass

            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                futures = {pool.submit(_run_gpu_call, c): c for c in batch}
                for future in as_completed(futures):
                    t_name, t_args, success, summary, raw_res = future.result()
                    log(f"[Decider] Parallel GPU tool '{t_name}' finished: {summary[:80]}")
                    _entry = {
                        "step": len(execution_log) + 1,
                        "tool": t_name,
                        "args": {k: v for k, v in t_args.items() if k != "thread_safe_state"},
                        "success": success,
                        "result_summary": summary,
                    }
                    execution_log.append(_entry)
                    _append_part4_step(ledger_path, _entry)  # crash-safe: persist immediately
                    execution_log_batch.append({
                        "step": len(execution_log),
                        "tool": t_name,
                        "success": success,
                        "result_raw": json.dumps(raw_res, default=str),
                    })
                    steps_remaining -= 1

        # Append all tool results from this batch to the conversation history
        if execution_log_batch:
            results_text_parts = []
            for entry in execution_log_batch:
                success_str = "SUCCESS" if entry["success"] else "FAILED"
                results_text_parts.append(
                    f"Result of step {entry['step']} (`{entry['tool']}`): {success_str}\n"
                    f"<tool_result>\n{entry['result_raw']}\n</tool_result>"
                )
            results_prompt = "\n\n".join(results_text_parts)
            conversation_history.append({"role": "user", "parts": [results_prompt]})

    # --- Close Part 4 (header + steps already written incrementally) ---
    _write_part4_footer(ledger_path, execution_log, total_budget, halt_reason)
    log(f"[Decider] Execution complete. {len(execution_log)} steps taken, "
        f"{steps_remaining} remaining. Part 4 written.")
    return True


def _format_part4_step(entry: Dict) -> str:
    """Format a single execution step as Part 4 markdown (pure, no I/O)."""
    status = "OK" if entry["success"] else "FAILED"
    lines = [f"**Step {entry['step']}:** `{entry['tool']}` [{status}]"]
    # Args: show full hyperparameter dict. Only truncate if extremely large (>2000 chars)
    # so the Theorist can see exactly what was tried each step.
    if entry.get("args"):
        args_str = json.dumps(entry["args"], default=str)
        if len(args_str) > 2000:
            args_str = args_str[:2000] + "...(truncated — see results_log.json for full params)"
        lines.append(f"  - Args: `{args_str}`")
    # Result: never truncate error messages — full text is required for next-cycle diagnosis
    result_text = entry["result_summary"]
    lines.append(f"  - Result: {result_text}")
    # Extra debug block for failures so they are impossible to miss
    if not entry["success"]:
        lines.append(f"  - ⚠️ FAILURE DETAIL: {result_text}")
    lines.append("")
    return "\n".join(lines)


def _format_part4_header(total_budget: int) -> str:
    """Part 4 opening (written once, before the execution loop)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return "\n".join([
        "## Part 4: Decider's Execution Log",
        f"**Started:** {timestamp}",
        f"**Budget:** {total_budget} steps planned",
        "",
        "### Execution Steps",
        "",
    ])


def _format_part4_footer(
    execution_log: List[Dict], total_budget: int, halt_reason: Optional[str] = None
) -> str:
    """Part 4 closing summary (written once, after the loop)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    steps_used = len(execution_log)
    successful = sum(1 for e in execution_log if e["success"])
    failed = steps_used - successful
    lines = [
        "### Summary",
        f"**Completed:** {timestamp}",
        f"**Budget:** {steps_used}/{total_budget} steps used",
        f"**Results:** {successful} succeeded, {failed} failed",
    ]
    if halt_reason:
        lines.append(f"**Halt Reason:** {halt_reason}")
    return "\n".join(lines)


def _write_part4_header(ledger_path: str, total_budget: int) -> None:
    """Open Part 4 as a new section (the only call that adds the section divider)."""
    append_section(ledger_path, _format_part4_header(total_budget))


def _append_part4_step(ledger_path: str, entry: Dict) -> None:
    """Append one step inside the open Part 4 section (append-only, crash-safe)."""
    append_raw(ledger_path, _format_part4_step(entry))


def _write_part4_footer(
    ledger_path: str, execution_log: List[Dict], total_budget: int, halt_reason: Optional[str] = None
) -> None:
    """Close Part 4 with the summary footer (inside the open section)."""
    append_raw(ledger_path, _format_part4_footer(execution_log, total_budget, halt_reason))


def _write_part4(
    ledger_path: str,
    execution_log: List[Dict],
    objectives: List[str],
    total_budget: int,
    halt_reason: Optional[str] = None,
) -> None:
    """One-shot Part 4 writer — used only for the no-objectives / no-steps early path.

    The main execution loop writes Part 4 incrementally (header → per-step → footer)
    so an interrupt never loses the record; see _write_part4_header / _append_part4_step.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    steps_used = len(execution_log)
    successful = sum(1 for e in execution_log if e["success"])
    failed = steps_used - successful

    lines = [
        f"## Part 4: Decider's Execution Log",
        f"**Date:** {timestamp}",
        f"**Budget:** {steps_used}/{total_budget} steps used",
        f"**Results:** {successful} succeeded, {failed} failed",
        "",
    ]

    if halt_reason:
        lines.append(f"**Halt Reason:** {halt_reason}")
        lines.append("")

    if execution_log:
        lines.append("### Execution Steps")
        lines.append("")
        for entry in execution_log:
            lines.append(_format_part4_step(entry))
    else:
        lines.append("(No steps executed)")
        lines.append("")

    append_section(ledger_path, "\n".join(lines))
