# epoch_runner.py
"""
Single Ledger Architecture — Phase 3: Epoch Runner.

The epoch runner is the top-level autonomous loop that orchestrates:
    Analyst (Part 1) → Theorist (Parts 2-3) → Decider (Part 4)

Each full cycle = 1 epoch = 1 ledger file.

The epoch runner reads user_directive.md at every cycle boundary
for pause/resume, priority overrides, and budget constraints.

Note: This module can be used as the sole entry point once the
legacy orchestration (run_agent_turn + DeciderOrchestrator) is retired.
Currently it coexists with the existing system via main.py hooks.
"""

import time
from typing import Any, Callable, Dict, Optional

from ledger_schema import create_ledger, get_latest_ledger, read_ledger, SECTION_MARKERS
from analyst import run_analyst
from theorist import run_theorist
from decider_ledger import run_decider
from summarizer_ledger import run_multi_agent_summarizer
from user_directive import read_user_directive, UserDirective
from ui_logger import log


def run_single_epoch(
    thread_safe_state: Dict[str, Any],
    execute_tool_func: Callable,
    workspace_dir: str = "workspace/",
    experiment_name: str = "Agent_Brain",
    preamble: Optional[str] = None,
    directive: Optional[UserDirective] = None,
) -> tuple[bool, Optional[str]]:
    """
    Run a single Analyst → Theorist → Decider epoch.

    This is the atomic unit of the Single Ledger Architecture.
    Can be called from main.py's loop or from run_epoch_loop().

    Args:
        directive: Pre-loaded UserDirective. When supplied by run_epoch_loop the
            file is NOT re-read, eliminating the race condition where the user
            edits the file between the loop check and cycle execution.
            If None, the directive is read fresh from disk (default behaviour
            when called standalone).

    Returns True if the cycle completed (even partially), False on critical failure.
    """
    # 1. Read user directive (only if not passed in from the loop to avoid double-read)
    if directive is None:
        directive = read_user_directive(workspace_dir)

    if directive.status == "paused":
        log("[EpochRunner] System paused by user directive.")
        return False, None

    if directive.status == "stopped":
        log("[EpochRunner] System stopped by user directive.")
        return False, None

    # 2. Create new cycle ledger
    prev_cycle_num, prev_ledger_path = get_latest_ledger(workspace_dir)
    next_cycle_num = prev_cycle_num + 1

    # Check epoch limits
    if directive.max_epochs and next_cycle_num > directive.max_epochs:
        log(f"[EpochRunner] Max epochs ({directive.max_epochs}) reached. Stopping.")
        return False, None

    if directive.pause_after_epoch and next_cycle_num > directive.pause_after_epoch:
        log(f"[EpochRunner] Pause point (epoch {directive.pause_after_epoch}) reached.")
        return False, None

    # Phase 0 chronicle removed: the Research Chronicle is no longer synthesized every cycle
    # nor injected as Part 0. It is now a standalone, pull-based technical report
    # (workspace/research_chronicle.md) generated once at session start and on Theorist request
    # (refresh_chronicle tool), consumed only by the Theorist. Part 0 carries only the legacy
    # preamble if one was passed in.
    part_0 = preamble

    ledger_path = create_ledger(next_cycle_num, workspace_dir, preamble=part_0)
    log(f"[EpochRunner] === Cycle {next_cycle_num} ===")

    # Phase 1: ANALYST — writes Part 1
    log(f"[EpochRunner] Phase 1: Analyst...")
    try:
        analyst_ok = run_analyst(
            ledger_path=ledger_path,
            thread_safe_state=thread_safe_state,
            prev_ledger_path=prev_ledger_path,
            experiment_name=experiment_name,
            user_notes=directive.notes,
        )
        if not analyst_ok:
            log("[EpochRunner] Analyst failed. Cycle aborted.")
            return False, None
    except Exception as e:
        log(f"[EpochRunner] Analyst exception: {e}")
        return False, None

    # Phase 2: THEORIST — writes Parts 2-3
    log(f"[EpochRunner] Phase 2: Theorist...")
    try:
        run_theorist(
            ledger_path=ledger_path,
            thread_safe_state=thread_safe_state,
            priority_override=directive.priority_override,
            constraints=directive.constraints,
            notes=directive.notes,
        )
    except Exception as e:
        log(f"[EpochRunner] Theorist exception: {e}")

    # Phase 3: DECIDER — checkpoint-based: runs if Part 3 exists in ledger
    log(f"[EpochRunner] Phase 3: Decider...")
    try:
        ledger_content = read_ledger(ledger_path)
        if SECTION_MARKERS["part_3"] in ledger_content:
            decider_ok = run_decider(
                ledger_path=ledger_path,
                thread_safe_state=thread_safe_state,
                execute_tool_func=execute_tool_func,
                constraints=directive.constraints,
                prev_ledger_path=prev_ledger_path,
            )
            if decider_ok:
                log(f"[EpochRunner] === Cycle {next_cycle_num} execution complete ===")
            else:
                log(f"[EpochRunner] Decider failed. Cycle {next_cycle_num} partial.")
        else:
            log("[EpochRunner] Part 3 not found — Decider skipped.")
    except Exception as e:
        log(f"[EpochRunner] Decider exception: {e}")

    # Phase 4: SUMMARIZER — distill cycle into appendix
    log(f"[EpochRunner] Phase 4: Summarizer...")
    try:
        run_multi_agent_summarizer(
            ledger_path=ledger_path,
            thread_safe_state=thread_safe_state,
            workspace_dir=workspace_dir,
        )
    except Exception as e:
        log(f"[EpochRunner] Summarizer exception: {e}")

    return True, None


def run_epoch_loop(
    thread_safe_state: Dict[str, Any],
    execute_tool_func: Callable,
    workspace_dir: str = "workspace/",
    experiment_name: str = "Agent_Brain",
    poll_interval: int = 30,
) -> None:
    """
    Main autonomous loop: Analyst → Theorist → Decider, repeat.

    Reads user_directive.md at every cycle boundary.
    Pauses when directive says "paused", stops on "stopped" or max_epochs.

    Args:
        thread_safe_state: Shared state proxy.
        execute_tool_func: Tool dispatcher (main.py's execute_tool).
        workspace_dir: Where ledger files and user_directive.md live.
        experiment_name: MLflow experiment name for Analyst queries.
        poll_interval: Seconds to wait when paused before re-checking directive.
    """
    log("[EpochRunner] Starting autonomous epoch loop...")

    # Session-start bootstrap: build the Research Chronicle ONCE (not per-cycle) and flag it for
    # delivery to the first Theorist cycle. Thereafter it is pull-based (the Theorist requests a
    # refresh via the refresh_chronicle tool). Standalone file — never injected into the ledger.
    try:
        from summarizer_ledger import build_research_chronicle
        if build_research_chronicle(thread_safe_state, workspace_dir):
            thread_safe_state["chronicle_deliver_next"] = True
            log("[EpochRunner] Research Chronicle bootstrapped at session start.")
    except Exception as e:
        log(f"[EpochRunner] Chronicle bootstrap skipped: {e}")

    while True:
        directive = read_user_directive(workspace_dir)

        if directive.status == "stopped":
            log("[EpochRunner] Status: stopped. Exiting loop.")
            break

        if directive.status == "paused":
            log(f"[EpochRunner] Status: paused. Waiting {poll_interval}s...")
            time.sleep(poll_interval)
            continue

        # In-cycle halt request set by deep-stack tools (e.g. manifest_gate when
        # raw_data_path is unresolvable). Treated identically to Status:paused —
        # the user must clear the flag before the loop resumes.
        halt = thread_safe_state.get("halt_request") if hasattr(thread_safe_state, "get") else None
        if halt:
            reason = halt.get("reason", "unspecified") if isinstance(halt, dict) else str(halt)
            log(
                f"[EpochRunner] HALT REQUESTED ({reason}). Waiting {poll_interval}s for user to clear "
                f"thread_safe_state['halt_request']. Set Status:paused in directive if longer wait needed."
            )
            time.sleep(poll_interval)
            continue

        # Check epoch limit
        current_cycle, _ = get_latest_ledger(workspace_dir)
        if directive.max_epochs and current_cycle >= directive.max_epochs:
            log(f"[EpochRunner] Max epochs ({directive.max_epochs}) reached. Exiting.")
            break

        if directive.pause_after_epoch and current_cycle >= directive.pause_after_epoch:
            log(f"[EpochRunner] Pause-after-epoch ({directive.pause_after_epoch}) reached. Pausing.")
            time.sleep(poll_interval)
            continue

        # Run one epoch — pass the already-read directive to skip the internal
        # re-read and prevent seeing a partially-modified file mid-cycle.
        success, _ = run_single_epoch(
            thread_safe_state=thread_safe_state,
            execute_tool_func=execute_tool_func,
            workspace_dir=workspace_dir,
            experiment_name=experiment_name,
            directive=directive,
        )

        if not success:
            log("[EpochRunner] Epoch failed. Waiting before retry...")
            time.sleep(poll_interval)

    log("[EpochRunner] Epoch loop terminated.")
