# theorist.py
"""
Single Ledger Architecture — Phase 2: Theorist Deep Search Pipeline.

The Theorist reads Part 1 (Analyst's post-mortem) and produces Parts 2-3
(Strategic Context + Strategic Plan & Objectives) via a 4-stage checkpoint-
recoverable pipeline.

Contract: The Theorist never mutates state. It reads knowledge sources
(graph, RAG, memory) and writes to the ledger only.
"""

import json
import re
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from ledger_schema import (
    append_section,
    parse_section,
    read_ledger,
    SECTION_MARKERS,
)
from ui_logger import log
from cluster.prompts import (
    THEORIST_KNOWLEDGE_SCOUT_PROMPT,
    THEORIST_HYPOTHESIS_PROMPT,
    THEORIST_OBJECTIVE_EXPANSION_PROMPT,
    THEORIST_BUDGETING_PROMPT,
    THEORIST_STRATEGIC_CRITIC_PROMPT,
    THEORIST_HYPOTHESIS_REFINEMENT_PROMPT,
)


def _get_theorist_agent(thread_safe_state):
    """Return theorist_agent if configured, else fall back to main agent."""
    return (
        thread_safe_state.get("theorist_agent")
        or thread_safe_state.get("agent")
    )


# ---------------------------------------------------------------------------
# Checkpoint Stage Definition
# ---------------------------------------------------------------------------

@dataclass
class CheckpointStage:
    """A single stage in the Theorist pipeline with checkpoint recovery."""
    name: str
    marker: str           # String that must appear in ledger when stage is done
    runner: Callable      # Function(ledger_content, context) -> str
    max_retries: int = 3


class LLMCallError(Exception):
    """Raised when an LLM call fails in a Theorist stage."""
    pass


# ---------------------------------------------------------------------------
# Knowledge Gathering (read-only helpers)
# ---------------------------------------------------------------------------

def _gather_graph_data(thread_safe_state: Dict[str, Any]) -> str:
    """Read-only: get strategic graph summary for the Knowledge Scout."""
    try:
        graph_memory = thread_safe_state.get("strategic_graph_memory")
        if graph_memory is None:
            return "Strategic Graph: Not initialized."

        graph = graph_memory.graph
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        lines = [f"Strategic Graph: {num_nodes} nodes, {num_edges} edges."]

        # High-value states — show architecture label + Q-value only, no raw node IDs
        high_value = graph_memory.get_high_value_states(top_k=5, min_q_value=0.1)
        if high_value:
            labels = []
            for n in high_value:
                arch = graph_memory.get_node_arch(n)
                avg_q = graph_memory.get_node_avg_q(n)
                q_str = f"Q={avg_q:.3f}" if avg_q is not None else "Q=?"
                labels.append(f"{arch} ({q_str})")
            lines.append(f"High-Value States (top 5): {', '.join(labels)}")
        else:
            lines.append("High-Value States: None identified yet.")

        # Dead-end detection: nodes with many outgoing edges but low Q-values.
        # Skip when graph is too sparse to have meaningful dead-ends.
        try:
            if num_nodes >= 5:
                dead_ends = []
                for node in graph.nodes():
                    out_edges = list(graph.edges(node, data=True, keys=True))
                    if len(out_edges) >= 3:
                        q_values = [
                            data.get("impact_vector", [])[0]
                            for _, _, _, data in out_edges
                            if data.get("impact_vector")
                        ]
                        if q_values and max(q_values) < 0.01:
                            dead_ends.append(node)
                if dead_ends:
                    dead_end_labels = list(dict.fromkeys(
                        graph_memory.get_node_arch(n) for n in dead_ends[:5]
                    ))
                    lines.append(f"Potential Dead-End Architectures: {', '.join(dead_end_labels)}")
        except Exception:
            pass  # Non-critical

        return "\n".join(lines)

    except Exception as e:
        return f"Strategic Graph: Error reading ({e})."


def _gather_memory_data(thread_safe_state: Dict[str, Any], query: str) -> str:
    """Read-only: recall relevant memories for the Knowledge Scout.
    Fetches k=7 findings and compresses them via LLM summarizer to keep tokens low."""
    try:
        from tools import recall_relevant_memories
        result = recall_relevant_memories(
            query=query,
            thread_safe_state=thread_safe_state,
            k=7,
            summarize=True,  # LLM summarizer compresses findings — token-safe
        )
        if result.get("status") == "completed":
            memories = result.get("memories", [])
            if not memories:
                return "Long-Term Memory: No relevant memories found."
            # summarize=True returns a single string from the LLM summarizer
            if isinstance(memories, str):
                return f"Long-Term Memory Summary:\n{memories}"
            # Fallback if summarizer returned a list (e.g. agent unavailable)
            lines = ["Long-Term Memory Recall:"]
            for m in memories[:7]:
                if isinstance(m, dict):
                    lines.append(f"  - {m.get('finding', 'N/A')[:300]}")
                else:
                    lines.append(f"  - {str(m)[:300]}")
            return "\n".join(lines)
        return f"Long-Term Memory: {result.get('message', 'No results')}"
    except Exception as e:
        return f"Long-Term Memory: Error ({e})."


def _gather_architecture_code(thread_safe_state: Dict[str, Any]) -> str:
    """Read source code of the current custom architecture file, if any.
    Only called when selected_model looks like a .py file (custom arch).
    Truncated to 6000 chars to avoid context bloat."""
    selected = thread_safe_state.get("selected_model", "")
    if not selected.endswith(".py"):
        return "(No custom architecture file active — using standard builder.)"
    try:
        from tools import read_file_from_architectures
        result = read_file_from_architectures(filename=selected)
        if result.get("status") == "completed":
            code = result.get("content", "")
            if len(code) > 6000:
                code = code[:6000] + "\n... [truncated — first 6000 chars shown]"
            log(f"[Theorist/Scout] Loaded architecture source: {selected} ({len(code)} chars).")
            return f"```python\n# {selected}\n{code}\n```"
        return f"(Could not read {selected}: {result.get('message', 'unknown error')})"
    except Exception as e:
        return f"(Architecture read error: {e})"


def _gather_available_datasets(thread_safe_state: Dict[str, Any]) -> str:
    """Return a compact summary of already-generated datasets and representations.
    Injected into Theorist prompts so it never proposes generating data that exists."""
    try:
        from tools import list_available_datasets
        result = list_available_datasets(thread_safe_state=thread_safe_state)
        datasets = result.get("datasets", [])
        if not datasets:
            return "(No datasets found on disk.)"
        lines = []
        for ds in datasets:
            name = ds.get("manifest_name") or ds.get("name", "?")
            reps = ds.get("representations", ds.get("available_representations", []))
            reps_str = ", ".join(reps) if reps else "none"
            lines.append(f"- {name}: {reps_str}")
        return "\n".join(lines)
    except Exception as e:
        return f"(Could not list datasets: {e})"


def _gather_research_data(thread_safe_state: Dict[str, Any], query: str) -> str:
    """Read-only: query RAG archive for the Knowledge Scout."""
    try:
        from tools import query_research_archive
        result = query_research_archive(
            query=query,
            top_k=3,
            thread_safe_state=thread_safe_state,
        )
        if result.get("status") == "completed":
            chunks = result.get("results", [])
            if not chunks:
                return "Research Archive: No relevant documents found."
            lines = ["Research Archive Findings:"]
            for c in chunks[:3]:
                text = c.get("text", "N/A")[:200]
                source = c.get("source", "unknown")
                lines.append(f"  - [{source}] {text}")
            return "\n".join(lines)
        return f"Research Archive: {result.get('message', 'No results')}"
    except Exception as e:
        return f"Research Archive: Error ({e})."


def _build_memory_query(part_1_content: str, thread_safe_state: Dict[str, Any] = None) -> str:
    """Build a semantic search query from current context + Part 1 math facts.
    Including architecture and dataset makes the vector search much more targeted."""
    parts = []
    # Architecture and dataset give semantic search the most relevant anchor
    if thread_safe_state:
        model = thread_safe_state.get("selected_model", "")
        dataset = thread_safe_state.get("raw_data_source", "") or thread_safe_state.get("selected_raw_data_source", "")
        if model:
            parts.append(model)
        if dataset:
            parts.append(dataset)
    # Performance anchors from the deterministic Decision Briefing (A.1).
    # Part 1 is no longer a "Math Fact" list — pull the Performance & Gap bullets.
    perf_match = re.search(
        r"##\s*A\.1 PERFORMANCE & GAP(.*?)(?=\n##\s|\Z)", part_1_content, re.DOTALL
    )
    if perf_match:
        perf_lines = [
            ln.strip().lstrip("-").strip()
            for ln in perf_match.group(1).splitlines()
            if ln.strip().startswith("-")
        ]
        parts.extend(perf_lines[:3])
    else:
        # Legacy fallback: old "Math Fact N" ledger format.
        facts = re.findall(r"\*\*Math Fact \d+:\*\*\s*(.+)", part_1_content)
        parts.extend(facts[:2])
    if parts:
        return ". ".join(parts)
    # Final fallback: first 150 chars
    return part_1_content[:150]


def _part1_decision_slice(part_1_content: str, max_chars: int = 1800) -> str:
    """Decision-relevant slice of Part 1 for token-bounded downstream prompts.

    Part 1 is now a deterministic Decision Briefing (A.x) + an LLM Interpretation
    block (B). Instead of a blind char cut (which bisects a markdown table), return
    the Performance & Gap header (A.1) + the Interpretation & Flags section (B) —
    the parts a critic actually needs. Falls back to a head slice for legacy ledgers.
    """
    def _section(name: str) -> str:
        m = re.search(rf"(##\s*{re.escape(name)}.*?)(?=\n##\s|\Z)", part_1_content, re.DOTALL)
        return m.group(1).strip() if m else ""

    perf = _section("A.1 PERFORMANCE & GAP")
    interp = _section("B. INTERPRETATION & FLAGS")
    slice_parts = [s for s in (perf, interp) if s]
    if not slice_parts:
        return part_1_content[:max_chars]
    return "\n\n".join(slice_parts)[:max_chars]


def _gather_strategic_path(thread_safe_state: Dict[str, Any]) -> str:
    """Read-only: get A* path recommendation from the strategic graph (MOSAN).
    Only called when strategic_advisor_enabled=True."""
    try:
        from tools import get_strategic_path
        result = get_strategic_path(thread_safe_state=thread_safe_state)
        if result.get("status") != "completed":
            return f"Strategic Path: {result.get('message', 'Unavailable.')}"
        path = result.get("path", [])
        if not path:
            return "Strategic Path: No viable path found (graph too sparse — continue exploration)."
        improvement = result.get("expected_improvement", 0)
        lines = [f"Strategic Path (A*, expected +{improvement:.1%} improvement):"]
        for step in path[:5]:  # Show first 5 steps max
            conf = step.get("confidence", "?")
            if conf == "low":
                continue  # Skip low-confidence steps
            action = step.get("action", "?")
            gain = step.get("expected_gain", 0)
            lines.append(f"  Step {step['step']}: {action} (gain={gain:.4f}, conf={conf})")
        return "\n".join(lines) if len(lines) > 1 else "Strategic Path: All steps low-confidence — explore freely."
    except Exception as e:
        return f"Strategic Path: Error ({e})."


def _gather_mosan_advice(thread_safe_state: Dict[str, Any]) -> str:
    """
    Synchronously extracts Pareto-ranked advice from MOSAN (ContextCriticAgent + StrategicAdvisorTeam).
    Reads the 'default_lens' from config to guide the advisor.
    """
    try:
        from context_planner import ContextCriticAgent, StrategicAdvisorTeam
        from memory_utils import create_vector_memory
        import asyncio
        from omegaconf import OmegaConf
        import os

        # Load config to get default_lens and per-lens saturation settings
        project_root = os.path.abspath(os.path.dirname(__file__))
        config_path = os.path.join(project_root, "config.yaml")
        default_lens = "auto"
        saturation_config = None
        if os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
            sa_cfg = cfg.get("strategic_advisor", {}) or {}
            default_lens = sa_cfg.get("default_lens", "auto")
            # Build saturation_config dict (None when feature disabled).
            if sa_cfg.get("saturation_enabled", False):
                saturation_config = {
                    "saturation_enabled": True,
                    "saturation_min_graph_nodes": sa_cfg.get("saturation_min_graph_nodes", 10),
                    "saturation_recent_window": sa_cfg.get("saturation_recent_window", 10),
                    "explorer_novelty_percentile": sa_cfg.get("explorer_novelty_percentile", 0.75),
                    "hawk_accuracy_slope_threshold": sa_cfg.get("hawk_accuracy_slope_threshold", 0.001),
                    "sage_stability_std_threshold": sa_cfg.get("sage_stability_std_threshold", 0.01),
                    "engineer_coverage_threshold": sa_cfg.get("engineer_coverage_threshold", 0.8),
                }

        agent = _get_theorist_agent(thread_safe_state)
        if not agent:
            return "MOSAN Advisor: No agent found in state."

        graph_memory = thread_safe_state.get("strategic_graph_memory")
        if not graph_memory:
            return "MOSAN Advisor: No graph memory found."

        # Vector memory
        vector_memory = create_vector_memory(backend="sqlite", fallback_to_json=True)

        critic = ContextCriticAgent(agent)
        advisor = StrategicAdvisorTeam(agent, graph_memory, vector_memory, saturation_config=saturation_config)

        async def _run_mosan_logic():
            context_summary = await critic.summarize_context(thread_safe_state, graph_memory)
            advice_string = await advisor.generate_advice(
                context_summary=context_summary,
                thread_safe_state=thread_safe_state,
                default_lens=default_lens
            )
            return advice_string

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _run_mosan_logic())
                advice = future.result(timeout=60)
        else:
            advice = asyncio.run(_run_mosan_logic())
        return advice

    except Exception as e:
        import traceback
        return f"MOSAN Advisor Error ({e})\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Stage Runners — each returns the markdown content to append to the ledger
# ---------------------------------------------------------------------------

def _run_knowledge_scout(
    ledger_content: str,
    thread_safe_state: Dict[str, Any],
    priority_override: Optional[str] = None,
    constraints: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Stage 0: Knowledge Scout — gathers strategic context from graph, memory, RAG.
    Returns Part 2 content to append.
    """
    part_1 = parse_section_from_content(ledger_content, "Part 1: Analytical Post-Mortem")
    if not part_1:
        raise LLMCallError("Cannot run Knowledge Scout: Part 1 not found in ledger.")

    # Build search query from Part 1 content
    search_query = _build_memory_query(part_1, thread_safe_state)

    # Gather read-only intelligence
    graph_data = _gather_graph_data(thread_safe_state)
    memory_data = _gather_memory_data(thread_safe_state, search_query)
    research_data = _gather_research_data(thread_safe_state, search_query)
    architecture_code = _gather_architecture_code(thread_safe_state)
    available_datasets = _gather_available_datasets(thread_safe_state)

    # --- Phase-3 -> Theorist feed-forward (one-shot) ---
    # Knowledge the Decider produced/read last cycle (web report, explored architectures)
    # is otherwise lost to the Theorist. Inject the compact carryover directly, then clear.
    _web = thread_safe_state.get("last_web_report")
    if isinstance(_web, dict) and _web.get("summary"):
        research_data = (
            f"WEB RESEARCH REPORT (commissioned last cycle for: '{_web.get('query', '')}'):\n"
            f"{_web.get('summary')}\n[full report on disk: {_web.get('filepath', '?')}]\n\n"
            + (research_data or "")
        )
        thread_safe_state["last_web_report"] = None
    _arch = thread_safe_state.get("last_arch_insight")
    if isinstance(_arch, dict) and _arch.get("snippet"):
        architecture_code = (
            f"# --- ARCHITECTURE EXPLORED BY THE DECIDER LAST CYCLE: {_arch.get('filename')} (head) ---\n"
            f"{_arch.get('snippet')}\n\n"
            + (architecture_code or "")
        )
        thread_safe_state["last_arch_insight"] = None

    # MOSAN strategic path — only when toggle is enabled
    strategic_path_data = (
        _gather_strategic_path(thread_safe_state)
        if thread_safe_state.get("strategic_advisor_enabled", False)
        else "MOSAN disabled (strategic_advisor: enabled: false)."
    )
    # MOSAN Multi-Lens Pareto Advice
    mosan_advice_data = (
        _gather_mosan_advice(thread_safe_state)
        if thread_safe_state.get("strategic_advisor_enabled", False)
        else ""
    )

    # Build prompt
    metadata_info = (
        f"{thread_safe_state.get('selected_model', 'unknown')} "
        f"on {thread_safe_state.get('raw_data_source', 'unknown dataset')}"
    )
    prompt = THEORIST_KNOWLEDGE_SCOUT_PROMPT.format(
        part_1_content=part_1,
        graph_data=graph_data,
        memory_data=memory_data,
        research_data=research_data,
        metadata_info=metadata_info,
        strategic_path_data=strategic_path_data,
        mosan_advice_data=mosan_advice_data,
        architecture_code=architecture_code,
    )

    # Prepend user directive block as the FIRST thing the Scout sees.
    # Hierarchy: USER (priority_override + constraints + notes) > Theorist > Chronicler
    # Any of the three may be None — only non-empty sections are injected.
    directive_lines = []

    # --- OPERATIONAL BUDGET LIMITS (Finetune Gate) ---
    cfg = thread_safe_state.get("cfg")
    _training_cfg = (cfg.get("training") if isinstance(cfg, dict) else getattr(cfg, "training", None)) if cfg else None
    _get_cfg = (lambda k, d: _training_cfg.get(k, d)) if isinstance(_training_cfg, dict) else (lambda k, d: getattr(_training_cfg, k, d)) if _training_cfg else (lambda k, d: d)
    finetune_cap = _get_cfg("finetune_session_cap", 2)
    finetune_calls = thread_safe_state.get("finetune_calls_this_session", 0)
    finetune_remaining = max(0, finetune_cap - finetune_calls) if finetune_cap > 0 else "unlimited"
    
    directive_lines.append(
        f"### OPERATIONAL BUDGET LIMITS (Finetune Gate)\n"
        f"- **Finetuning Session Cap:** {finetune_cap} calls max per session.\n"
        f"- **Finetuning Calls Made So Far:** {finetune_calls} call(s).\n"
        f"- **Remaining Finetuning Calls Allowed:** {finetune_remaining}.\n"
        f"CRITICAL: If Remaining is 0, you are strictly FORBIDDEN from proposing objectives that call `run_training_trial_with_finetune`. You must rely on standard `run_training_trial` instead."
    )
    # -------------------------------------------------
    if priority_override:
        directive_lines.append(
            f"### PRIORITY OVERRIDE (absolute — overrides all other sources)\n{priority_override.strip()}"
        )
    if constraints:
        constraints_text = "\n".join(f"- {c}" for c in constraints)
        directive_lines.append(
            f"### USER CONSTRAINTS (must be respected in your plan — USER > Theorist > Chronicler)\n{constraints_text}"
        )
    if notes:
        directive_lines.append(
            f"### USER NOTES (context and awareness — lower priority than constraints)\n{notes.strip()}"
        )
    directive_lines.append(
        f"### AVAILABLE DATASETS ON DISK (do NOT propose generating these — they already exist)\n{available_datasets}"
    )
    if directive_lines:
        log(f"[Theorist/Scout] Injecting user directive block ({len(directive_lines)} section(s)).")
        prompt = (
            f"## USER DIRECTIVE (from operator — highest authority)\n"
            + "\n\n".join(directive_lines)
            + "\n\n---\n\n"
        ) + prompt

    # Call LLM (air-gapped, no conversation context)
    agent = _get_theorist_agent(thread_safe_state)
    if agent is None:
        raise LLMCallError("No agent found in state.")

    response_text, _ = agent.run_turn(
        system_prompt="You are the Knowledge Scout for the Theorist pipeline. Output strategic context only.",
        conversation_history=[{"role": "user", "parts": [prompt]}],
    )

    if not response_text or not response_text.strip():
        raise LLMCallError("Knowledge Scout: LLM returned empty response.")

    # Format Part 2 — prepend USER DIRECTIVE block so all downstream stages
    # (Hypothesis, Budgeting, Decider) can read it from the ledger directly.
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    directive_block = ""
    if priority_override or constraints or notes:
        sections = []
        if priority_override:
            sections.append(f"**PRIORITY OVERRIDE:** {priority_override.strip()}")
        if constraints:
            clist = "\n".join(f"- {c}" for c in constraints)
            sections.append(f"**CONSTRAINTS:**\n{clist}")
        if notes:
            sections.append(f"**NOTES:** {notes.strip()}")
        directive_block = (
            f"## USER DIRECTIVE (operator — highest authority: USER > Theorist > Chronicler)\n"
            + "\n\n".join(sections)
            + "\n\n---\n\n"
        )

    return (
        f"{directive_block}"
        f"## Part 2: Strategic Context\n"
        f"**Date:** {timestamp}\n\n"
        f"{response_text.strip()}"
    )


def _compute_lens_ranked_data(
    results_log: list,
    selected_model: str,
    available_next_archs: list,
    lens: str,
) -> tuple:
    """
    Computes lens-specific ranked data from trial history.
    Returns (lens_ranked_data_block: str, reordered_available_next_archs: list).

    The data block is pre-ranked numeric evidence — NOT instructions.
    The Theorist reads what's at the top of the list and proposes accordingly.
    """
    import numpy as np
    from collections import defaultdict

    HO_KEY = "held_out_test_accuracy"
    INF_KEY = "inference_time_ms"

    # Group all trials by architecture
    by_arch: dict = defaultdict(list)
    for t in results_log:
        arch = t.get("architecture", "unknown")
        ho = t.get(HO_KEY)
        if ho is not None:
            by_arch[arch].append(t)

    # Build per-arch summary stats
    arch_stats = {}
    for arch, trials in by_arch.items():
        hos = [t[HO_KEY] for t in trials]
        infs = [t.get(INF_KEY, 0.0) for t in trials if t.get(INF_KEY) is not None]
        arch_stats[arch] = {
            "n": len(trials),
            "peak_ho": max(hos),
            "mean_ho": float(np.mean(hos)),
            "std_ho": float(np.std(hos)) if len(hos) > 1 else 0.0,
            "mean_inf": float(np.mean(infs)) if infs else 999.0,
            "recent_ho": hos[-1] if hos else 0.0,
        }

    lens = (lens or "auto").lower()

    def _sort_key_hawk(arch):
        s = arch_stats.get(arch, {})
        return -s.get("peak_ho", 0.0)

    def _sort_key_sage(arch):
        s = arch_stats.get(arch, {})
        return s.get("std_ho", 999.0)

    def _sort_key_explorer(arch):
        s = arch_stats.get(arch, {})
        return s.get("n", 0)

    def _sort_key_engineer(arch):
        s = arch_stats.get(arch, {})
        ho = s.get("mean_ho", 0.001) or 0.001
        inf = s.get("mean_inf", 999.0)
        return inf / ho  # lower = more efficient per unit accuracy

    sort_fn = {
        "hawk": _sort_key_hawk,
        "sage": _sort_key_sage,
        "explorer": _sort_key_explorer,
        "engineer": _sort_key_engineer,
    }.get(lens)

    reordered = list(available_next_archs)
    if sort_fn is not None:
        reordered = sorted(reordered, key=lambda a: sort_fn(a))

    # Build ranked data table for current model (last 10 trials only)
    current_trials = [t for t in results_log if t.get("architecture") == selected_model][-10:]

    rows = []
    for i, t in enumerate(current_trials):
        ho = t.get(HO_KEY, "N/A")
        inf = t.get(INF_KEY, "N/A")
        ho_str = f"{ho:.4f}" if isinstance(ho, float) else str(ho)
        inf_str = f"{inf:.1f}ms" if isinstance(inf, float) else str(inf)
        rows.append(f"  #{i+1}: HO={ho_str} | inf={inf_str}")

    # Cross-arch summary sorted by lens
    cross_rows = []
    all_archs_sorted = sorted(arch_stats.keys(), key=lambda a: sort_fn(a) if sort_fn else 0)
    for arch in all_archs_sorted[:6]:
        s = arch_stats[arch]
        cross_rows.append(
            f"  {arch}: n={s['n']} | peak_HO={s['peak_ho']:.4f} | std={s['std_ho']:.4f} | mean_inf={s['mean_inf']:.1f}ms"
        )

    lens_label = lens.upper() if lens != "auto" else "AUTO"
    block_lines = [
        f"## LENS={lens_label} RANKED SIGNAL (pre-sorted by lens priority — top = most relevant)",
        f"# Current arch ({selected_model}) — last {len(current_trials)} trials:",
        *rows,
        f"# Cross-arch ranking (sorted by {lens_label} criterion):",
        *cross_rows,
        f"# Next-arch candidates (ordered by {lens_label} priority):",
        f"  {', '.join(reordered) if reordered else 'none remaining'}",
    ]

    return "\n".join(block_lines), reordered


def _load_latest_critic_warning() -> str:
    """Read the most recent ledger appendix and extract its **Critique:** block.

    Mirrors decider_ledger.py so the Theorist sees the SAME structural warning the
    Decider already gets — closing the gap where code-bug alarms (compilation errors,
    broken pipelines) raised by the Critic never reached hypothesis generation.
    """
    try:
        import os as _os
        import re as _re
        from ledger_schema import WORKSPACE_DIR as _WS
        files = sorted(
            [f for f in _os.listdir(_WS) if _re.match(r"ledger_appendix_\d+\.md", f)],
            key=lambda x: int(_re.search(r"(\d+)", x).group(1)),
        )
        if not files:
            return "(no critic warning yet — first cycle)"
        with open(_os.path.join(_WS, files[-1]), "r", encoding="utf-8") as f:
            content = f.read()
        m = _re.search(r"\*\*Critique:\*\*(.*?)(?=\n#|\Z)", content, _re.DOTALL)
        warning = (m.group(1).strip() if m else content.strip())[:600]
        return warning or "(no critic warning yet)"
    except Exception:
        return "(no critic warning available)"


def _run_hypothesis_generation(
    ledger_content: str,
    thread_safe_state: Dict[str, Any],
    priority_override: Optional[str] = None,
    constraints: Optional[list] = None,
    notes: Optional[str] = None,
) -> str:
    """
    Stage 1: Hypothesis Generation — produces 1-3 falsifiable hypotheses.
    Returns the Hypothesis block to append.
    """
    part_1 = parse_section_from_content(ledger_content, "Part 1: Analytical Post-Mortem")
    part_2 = parse_section_from_content(ledger_content, "Part 2: Strategic Context")

    if not part_1 or not part_2:
        raise LLMCallError("Cannot generate hypotheses: Part 1 or Part 2 missing.")

    # Research Chronicle: PULL-based. Delivered to the Theorist ONLY the cycle after a refresh
    # was requested (the refresh_chronicle tool sets chronicle_deliver_next), or once at session
    # start (bootstrap in main.py). It is a standalone technical report
    # (workspace/research_chronicle.md), never written into the ledger.
    if thread_safe_state.get("chronicle_deliver_next"):
        try:
            import os as _os
            from ledger_schema import WORKSPACE_DIR as _WS
            _cpath = _os.path.join(_WS, "research_chronicle.md")
            if _os.path.exists(_cpath):
                with open(_cpath, "r", encoding="utf-8") as _f:
                    _chron = _f.read().strip()
                if _chron:
                    part_2 = f"{_chron}\n\n" + (part_2 or "")
        except Exception:
            pass
        thread_safe_state["chronicle_deliver_next"] = False

    # Build completed/abandoned architecture list to block the Theorist from re-proposing them.
    arch_flags = thread_safe_state.get("architecture_flags", {})
    closed_archs = [name for name, flag in arch_flags.items() if flag in ("completed", "abandoned_underperforming")]
    if closed_archs:
        completed_architectures = "- " + "\n- ".join(closed_archs)
    else:
        completed_architectures = "(none yet)"

    selected_model = thread_safe_state.get("selected_model", "unknown")
    raw_data_source = thread_safe_state.get("selected_raw_data_source", thread_safe_state.get("raw_data_source", "unknown dataset"))

    # Innovation context now comes from the on-disk architecture registry, which
    # surveys all 138+ forged architectures (pending / success / sys-failed),
    # filters by current dimensionality, and ranks by score. The legacy
    # "last cycle's innovation" string is preserved as the registry's tail block
    # for backward compatibility — see architecture_registry.build_innovation_context.
    try:
        from architecture_registry import build_innovation_context
        innovation_context = build_innovation_context(thread_safe_state, selected_model)
    except Exception as _reg_err:
        # Defensive fallback: never let the registry crash the Theorist.
        last_innovation = thread_safe_state.get("last_innovation_result")
        if last_innovation and isinstance(last_innovation, dict) and last_innovation.get("status") == "success":
            arch_file = last_innovation.get("architecture_file", "unknown")
            rep_type = last_innovation.get("representation_type", "unknown")
            innovation_context = (
                f"The Innovation Team successfully built a new architecture last cycle.\n"
                f"- File: `{arch_file}`\n"
                f"- Representation type: `{rep_type}`\n"
                f"**Your first objective MUST be to test and optimize this new architecture.**"
            )
        else:
            innovation_context = "(no new architecture built last cycle)"

    # --- Creative Team feed-forward (one-shot) ---
    # generate_creative_hypotheses runs in the Decider phase (after the Theorist has
    # already planned), so its chosen hypothesis can only influence the NEXT cycle.
    # Surface it here as a high-priority candidate, then clear it so it is not reused.
    creative = thread_safe_state.get("last_creative_hypothesis")
    if creative and isinstance(creative, dict) and creative.get("directive"):
        innovation_context = (
            "**CREATIVE TEAM DIRECTIVE (from last cycle — high priority: convert into an "
            "objective, or explicitly reject it with a reason):**\n"
            f"- action_type: {creative.get('action_type', 'unparsed')}\n"
            f"- directive: {creative.get('directive')}\n\n"
            + (innovation_context or "")
        )
        try:
            thread_safe_state["last_creative_hypothesis"] = None
        except Exception:
            pass

    # --- Compute escalation signals for the prompt ---
    results_log = thread_safe_state.get("results_log", [])
    arch_trials = [t for t in results_log if t.get("architecture") == selected_model]
    arch_trial_count = len(arch_trials)

    # Stagnation: slope of last 10 trials for current arch
    stagnation_label = "UNKNOWN (not enough trials)"
    if arch_trial_count >= 5:
        try:
            import numpy as np
            accs = [t.get("mean_accuracy", 0.0) for t in arch_trials[-10:]]
            slope = float(np.polyfit(range(len(accs)), accs, 1)[0])
            if abs(slope) < 0.0005:
                stagnation_label = f"STAGNANT (slope={slope:.5f})"
            elif slope > 0:
                stagnation_label = f"IMPROVING (slope={slope:.5f})"
            else:
                stagnation_label = f"DECLINING (slope={slope:.5f})"
        except Exception:
            stagnation_label = "UNKNOWN (computation error)"

    # Available representations for escalation candidates.
    # NOTE: these are DATA REPRESENTATIONS, not architectures.
    # The same representation can be paired with different architectures (standard CNN or custom .py file).
    # Flagging a representation as "abandoned" means the STANDARD CNN on that representation is exhausted,
    # NOT that a custom architecture on the same representation is also exhausted.
    all_representations = [
        "2D_GAF", "2D_CWT_SCALOGRAM",
        "3D_VIDEO", "3D_GAF_VIDEO", "3D_DYNAMIC_GAF",
        "3D_DYNAMIC_CWT", "3D_WAVELET_CWT",
    ]
    arch_flags = thread_safe_state.get("architecture_flags", {})
    available_next_archs = [r for r in all_representations if arch_flags.get(r) not in ("completed", "abandoned_underperforming")]

    # --- Lens: read active lens from config ---
    _active_lens = "auto"
    try:
        from omegaconf import OmegaConf
        import os as _os
        _cfg_path = _os.path.join(_os.path.abspath(_os.path.dirname(__file__)), "config.yaml")
        if _os.path.exists(_cfg_path):
            _cfg = OmegaConf.load(_cfg_path)
            _active_lens = (_cfg.get("strategic_advisor", {}) or {}).get("default_lens", "auto")
    except Exception:
        pass

    # Compute lens-ranked evidence and reorder arch candidates by lens priority
    lens_ranked_data, available_next_archs = _compute_lens_ranked_data(
        results_log=results_log,
        selected_model=selected_model,
        available_next_archs=available_next_archs,
        lens=_active_lens,
    )

    prompt = THEORIST_HYPOTHESIS_PROMPT.format(
        part_1_content=part_1,
        part_2_content=part_2,
        completed_architectures=completed_architectures,
        innovation_context=innovation_context,
        selected_model=selected_model,
        raw_data_source=raw_data_source,
        arch_trial_count=arch_trial_count,
        stagnation_label=stagnation_label,
        available_next_archs=", ".join(available_next_archs) if available_next_archs else "none remaining",
        lens_ranked_data=lens_ranked_data,
        critic_warning=_load_latest_critic_warning(),
    )

    # Prepend user directive — same hierarchy as knowledge_scout
    directive_lines = []
    if priority_override:
        directive_lines.append(
            f"### PRIORITY OVERRIDE (absolute — overrides all other sources)\n{priority_override.strip()}"
        )
    if constraints:
        directive_lines.append(
            f"### USER CONSTRAINTS\n" + "\n".join(f"- {c}" for c in constraints)
        )
    if notes:
        directive_lines.append(f"### USER NOTES\n{notes.strip()}")
    if directive_lines:
        prompt = (
            "## USER DIRECTIVE (highest authority — follow exactly)\n"
            + "\n\n".join(directive_lines)
            + "\n\n---\n\n"
        ) + prompt

    agent = _get_theorist_agent(thread_safe_state)
    if agent is None:
        raise LLMCallError("No agent found in state.")

    response_text, _ = agent.run_turn(
        system_prompt="You are a scientific hypothesis generator. Output only hypotheses.",
        conversation_history=[{"role": "user", "parts": [prompt]}],
    )

    if not response_text or not response_text.strip():
        raise LLMCallError("Hypothesis Generation: LLM returned empty response.")

    return f"**Hypothesis:**\n{response_text.strip()}"


def _run_hypothesis_debate(
    ledger_content: str,
    thread_safe_state: Dict[str, Any],
) -> str:
    """
    Stage 1b (conditional): Theorist ↔ Critic recursive refinement loop.

    Triggered only when stagnation is detected (STAGNANT or DECLINING trend,
    or arch_trial_count >= 15). In normal/improving runs this stage is skipped
    to avoid token cost.

    Two-round loop:
      Round 1 — Critic attacks the hypotheses, assigns Viability + Fatal Flaw.
      Round 2 — Refiner takes surviving hypotheses and generates 3 sub-variations
                 (Conservative / Aggressive / Balanced), selects one winner each.

    The refined hypotheses REPLACE the original hypothesis block in the ledger
    so that the Objective Expansion stage always sees the post-debate output.
    """
    hyp_block = _extract_block(ledger_content, "**Hypothesis:**")
    if not hyp_block:
        raise LLMCallError("Cannot run debate: Hypothesis block not found.")

    part_1 = parse_section_from_content(ledger_content, "Part 1: Analytical Post-Mortem") or ""

    selected_model = thread_safe_state.get("selected_model", "unknown")
    raw_data_source = thread_safe_state.get(
        "selected_raw_data_source",
        thread_safe_state.get("raw_data_source", "unknown dataset"),
    )
    results_log = thread_safe_state.get("results_log", [])
    arch_trials = [t for t in results_log if t.get("architecture") == selected_model]
    arch_trial_count = len(arch_trials)

    stagnation_label = "UNKNOWN"
    if arch_trial_count >= 5:
        try:
            import numpy as np
            accs = [t.get("mean_accuracy", 0.0) for t in arch_trials[-10:]]
            slope = float(np.polyfit(range(len(accs)), accs, 1)[0])
            if abs(slope) < 0.0005:
                stagnation_label = f"STAGNANT (slope={slope:.5f})"
            elif slope > 0:
                stagnation_label = f"IMPROVING (slope={slope:.5f})"
            else:
                stagnation_label = f"DECLINING (slope={slope:.5f})"
        except Exception:
            stagnation_label = "UNKNOWN"

    agent = _get_theorist_agent(thread_safe_state)
    if agent is None:
        raise LLMCallError("No agent found in state.")

    # --- Round 1: Critic attacks ---
    log("[Theorist/Debate] Round 1 — Strategic Critic reviewing hypotheses...")
    critic_prompt = THEORIST_STRATEGIC_CRITIC_PROMPT.format(
        selected_model=selected_model,
        raw_data_source=raw_data_source,
        stagnation_label=stagnation_label,
        arch_trial_count=arch_trial_count,
        hypothesis_block=hyp_block,
        part_1_content=_part1_decision_slice(part_1),  # Section-aware: A.1 + B, table-safe
    )
    critic_output, _ = agent.run_turn(
        system_prompt="You are a harsh scientific critic. Attack the hypotheses. Output structured critique only.",
        conversation_history=[{"role": "user", "parts": [critic_prompt]}],
    )
    if not critic_output or not critic_output.strip():
        raise LLMCallError("Hypothesis Debate: Critic returned empty response.")

    log(f"[Theorist/Debate] Critic output ({len(critic_output.split())} words).")

    # Extract surviving hypothesis labels from critic output
    surviving_match = re.search(r"\*\*SURVIVING HYPOTHESES:\*\*\s*(.+)", critic_output)
    surviving_labels = surviving_match.group(1).strip() if surviving_match else "H1"

    # If critic rejected everything, keep H1 as fallback
    if not surviving_labels or surviving_labels.lower() in ("none", ""):
        log("[Theorist/Debate] Critic rejected all hypotheses — keeping H1 as fallback.")
        surviving_labels = "H1"

    # --- Round 2: Refiner generates sub-variations and picks winner ---
    log("[Theorist/Debate] Round 2 — Refiner generating sub-variations...")
    refiner_prompt = THEORIST_HYPOTHESIS_REFINEMENT_PROMPT.format(
        selected_model=selected_model,
        raw_data_source=raw_data_source,
        stagnation_label=stagnation_label,
        hypothesis_block=hyp_block,
        critic_output=critic_output.strip(),
        surviving_labels=surviving_labels,
    )
    refined_output, _ = agent.run_turn(
        system_prompt="You are the Theorist Refiner. Generate sub-variations and select winners. Output structured refinement only.",
        conversation_history=[{"role": "user", "parts": [refiner_prompt]}],
    )
    if not refined_output or not refined_output.strip():
        raise LLMCallError("Hypothesis Debate: Refiner returned empty response.")

    log(f"[Theorist/Debate] Refined output ({len(refined_output.split())} words).")

    # Extract the REFINED HYPOTHESES block from refiner output
    refined_match = re.search(r"\*\*REFINED HYPOTHESES:\*\*(.+)", refined_output, re.DOTALL)
    if refined_match:
        refined_hyp_block = refined_match.group(1).strip()
    else:
        # Fallback: use full refiner output as the new hypothesis block
        refined_hyp_block = refined_output.strip()

    # Build the debate record to write to the ledger
    debate_record = (
        f"**Hypothesis Debate (Critic ↔ Refiner):**\n\n"
        f"### Critic Round\n{critic_output.strip()}\n\n"
        f"### Refiner Round\n{refined_output.strip()}\n\n"
        f"**Winning Hypothesis:**\n{refined_hyp_block}"
    )

    return debate_record


def _run_objective_expansion(
    ledger_content: str,
    thread_safe_state: Dict[str, Any],
) -> str:
    """
    Stage 2: Objective Expansion — converts each hypothesis into concrete objectives.
    One LLM call per hypothesis (max 3).
    Returns the Semantic Objectives block to append.
    """
    # Extract hypotheses — prefer the refined block from the debate stage if present,
    # otherwise fall back to the original hypothesis block.
    hyp_block = _extract_block(ledger_content, "**Winning Hypothesis:**")
    if not hyp_block:
        hyp_block = _extract_block(ledger_content, "**REFINED HYPOTHESES:**")
    if not hyp_block:
        hyp_block = _extract_block(ledger_content, "**Hypothesis:**")
    if not hyp_block:
        raise LLMCallError("Cannot expand objectives: Hypothesis block not found.")

    # Parse individual hypotheses (look for **H1:**, **H2:**, etc. or bullet points)
    hypotheses = re.findall(r"\*\*H\d+:\*\*\s*(.+)", hyp_block)
    if not hypotheses:
        # Fallback: try bullet points
        hypotheses = re.findall(r"[-•]\s*(.+)", hyp_block)
    if not hypotheses:
        # Last resort: treat entire block as one hypothesis
        hypotheses = [hyp_block.strip()]

    agent = _get_theorist_agent(thread_safe_state)
    if agent is None:
        raise LLMCallError("No agent found in state.")

    all_objectives = []
    obj_counter = 1

    selected_model = thread_safe_state.get("selected_model", "unknown")
    raw_data_source = thread_safe_state.get("selected_raw_data_source", thread_safe_state.get("raw_data_source", "unknown dataset"))

    for i, hypothesis in enumerate(hypotheses[:3]):  # Max 3 hypotheses
        prompt = THEORIST_OBJECTIVE_EXPANSION_PROMPT.format(
            hypothesis=hypothesis.strip(),
            selected_model=selected_model,
            raw_data_source=raw_data_source,
        )

        response_text, _ = agent.run_turn(
            system_prompt="You convert hypotheses into execution objectives. Output only objectives.",
            conversation_history=[{"role": "user", "parts": [prompt]}],
        )

        if response_text and response_text.strip():
            # Renumber objectives to be globally unique
            lines = response_text.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("- [ ]"):
                    # Replace O{N} with global counter
                    renumbered = re.sub(
                        r"O\d+:",
                        f"O{obj_counter}:",
                        line,
                        count=1,
                    )
                    all_objectives.append(renumbered)
                    obj_counter += 1
                elif line:
                    # If format is slightly off, still include it
                    all_objectives.append(f"- [ ] O{obj_counter}: {line}")
                    obj_counter += 1

    if not all_objectives:
        raise LLMCallError("Objective Expansion: No objectives generated.")

    return f"**Semantic Objectives:**\n" + "\n".join(all_objectives)


def _run_budgeting_assembly(
    ledger_content: str,
    thread_safe_state: Dict[str, Any],
) -> str:
    """
    Stage 3: Budgeting & Assembly — assigns step budget and writes Part 3.
    Returns the full Part 3 content to append.
    """
    obj_block = _extract_block(ledger_content, "**Semantic Objectives:**")
    if not obj_block:
        raise LLMCallError("Cannot budget: Semantic Objectives block not found.")

    global_step = thread_safe_state.get("global_step_counter", 0)
    results_log = thread_safe_state.get("results_log", [])
    selected_model = thread_safe_state.get("selected_model", "unknown")
    arch_trial_count = len([t for t in results_log if t.get("architecture") == selected_model])

    prompt = THEORIST_BUDGETING_PROMPT.format(
        all_objectives=obj_block,
        global_step=global_step,
        trial_count=arch_trial_count,
        selected_model=selected_model,
    )

    # --- CRITICAL OPERATIONAL CONSTRAINT ---
    cfg = thread_safe_state.get("cfg")
    _training_cfg = (cfg.get("training") if isinstance(cfg, dict) else getattr(cfg, "training", None)) if cfg else None
    _get_cfg = (lambda k, d: _training_cfg.get(k, d)) if isinstance(_training_cfg, dict) else (lambda k, d: getattr(_training_cfg, k, d)) if _training_cfg else (lambda k, d: d)
    finetune_cap = _get_cfg("finetune_session_cap", 2)
    finetune_calls = thread_safe_state.get("finetune_calls_this_session", 0)
    finetune_remaining = max(0, finetune_cap - finetune_calls) if finetune_cap > 0 else "unlimited"

    prompt += (
        f"\n\n### CRITICAL OPERATIONAL CONSTRAINT:\n"
        f"- Finetuning Session Cap: {finetune_cap} max calls per session.\n"
        f"- Finetuning Calls Already Made: {finetune_calls} call(s).\n"
        f"- Remaining Finetuning Calls: {finetune_remaining}.\n"
        f"CRITICAL: If Remaining is 0 or if the objectives propose more calls to `run_training_trial_with_finetune` "
        f"than {finetune_remaining}, you MUST reject/consolidate them or replace them with standard `run_training_trial` (1 step cost each). "
        f"Do NOT allocate more finetuning steps than the remaining limit!"
    )
    # ---------------------------------------

    agent = _get_theorist_agent(thread_safe_state)
    if agent is None:
        raise LLMCallError("No agent found in state.")

    response_text, _ = agent.run_turn(
        system_prompt="You are the Budget Controller. Output the budget and Part 3 in the exact format requested.",
        conversation_history=[{"role": "user", "parts": [prompt]}],
    )

    if not response_text or not response_text.strip():
        raise LLMCallError("Budgeting: LLM returned empty response.")

    # Validate that the response contains the required markers
    if "## Part 3:" not in response_text or "**Step Budget:**" not in response_text:
        raise LLMCallError("Budgeting: Response missing '## Part 3:' or '**Step Budget:**' marker.")

    # --- CRITICAL FIX: Restore verbatim objectives from Part 2 ---
    # The LLM paraphrases objectives, losing tool names and parameters.
    # We replace the LLM-written Semantic Objectives block with the exact
    # text from Part 2 (already validated by _run_objective_expansion).
    hyp_block = _extract_block(ledger_content, "**Winning Hypothesis:**")
    if not hyp_block:
        hyp_block = _extract_block(ledger_content, "**REFINED HYPOTHESES:**")
    if not hyp_block:
        hyp_block = _extract_block(ledger_content, "**Hypothesis:**")

    budget_match = re.search(r"\*\*Step Budget:\*\*\s*(\d+)", response_text)
    budget_val = budget_match.group(0) if budget_match else "**Step Budget:** 5"

    part3 = (
        "## Part 3: Strategic Plan & Objectives\n\n"
        "_(Winning hypothesis and full objectives are recorded in Part 2 — see "
        "`**Winning Hypothesis:**` and `**Semantic Objectives:**` blocks above.)_\n\n"
        f"{budget_val}"
    )

    return part3.strip()


# ---------------------------------------------------------------------------
# Helper: parse section from in-memory content (no file read)
# ---------------------------------------------------------------------------

def parse_section_from_content(content: str, section_name: str) -> Optional[str]:
    """Extract a section by its ## header from in-memory ledger content."""
    header_pattern = re.compile(
        rf"^## {re.escape(section_name)}\s*$", re.MULTILINE
    )
    match = header_pattern.search(content)
    if not match:
        return None

    start = match.end()
    next_section = re.compile(r"^---\s*$\s*^## ", re.MULTILINE)
    next_match = next_section.search(content, start)
    end = next_match.start() if next_match else len(content)

    return content[start:end].strip()


def _extract_block(content: str, marker: str) -> Optional[str]:
    """
    Extract content following a bold marker (e.g., **Hypothesis:**)
    until the next known section marker, section header, or end of content.
    Inline bold patterns like **H1:** are preserved (not treated as block ends).
    """
    idx = content.rfind(marker)
    if idx == -1:
        return None

    start = idx + len(marker)
    # Known block-ending markers — these signal a new logical section.
    # We do NOT end on **H\d+:** (hypothesis labels) or **O\d+:** (objective labels).
    block_end_markers = [
        SECTION_MARKERS.get("part_2", "## Part 2:"),
        SECTION_MARKERS.get("hypothesis", "**Hypothesis:**"),
        "**Winning Hypothesis:**",
        SECTION_MARKERS.get("objectives", "**Semantic Objectives:**"),
        SECTION_MARKERS.get("budget", "**Step Budget:**"),
        "**[THEORIST HALTED]**",
    ]
    # Remove the current marker from the end list so we don't self-terminate
    block_end_markers = [m for m in block_end_markers if m != marker]

    # Find the earliest known marker or ## header after start
    end = len(content)
    for m in block_end_markers:
        pos = content.find(m, start)
        if pos != -1 and pos < end:
            end = pos
    # Also check for ## headers and --- separators
    header_match = re.compile(r"^(---\s*$\s*^## )", re.MULTILINE).search(content, start)
    if header_match and header_match.start() < end:
        end = header_match.start()

    return content[start:end].strip()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_theorist(
    ledger_path: str,
    thread_safe_state: Dict[str, Any],
    priority_override: Optional[str] = None,
    constraints: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> bool:
    """
    Run the Theorist Deep Search pipeline with checkpoint recovery.

    Reads Part 1 from the ledger (written by Analyst), then runs 4 stages:
      0. Knowledge Scout  → writes Part 2: Strategic Context
      1. Hypothesis Gen   → writes **Hypothesis:** block
      2. Objective Expand  → writes **Semantic Objectives:** block
      3. Budgeting         → writes **Step Budget:** + full Part 3

    Each stage is checkpointed: if the ledger already contains the stage's
    marker, the stage is skipped. On failure, the pipeline halts and writes
    a [THEORIST HALTED] marker.

    Args:
        ledger_path: Path to the current cycle's ledger (Part 1 must exist).
        thread_safe_state: The shared state proxy (read-only access to graph, memory, etc.).
        priority_override: USER PRIORITY OVERRIDE — injected first, overrides everything.
        constraints: USER CONSTRAINTS — must be respected during planning (USER > Theorist > Chronicler).
        notes: USER NOTES — contextual awareness, lowest priority of the three.

    Returns:
        True if all stages completed successfully, False otherwise.
    """
    log("[Theorist] Starting Deep Search pipeline...")

    # Verify Part 1 exists before proceeding
    ledger_content = read_ledger(ledger_path)
    if SECTION_MARKERS["part_1"] not in ledger_content:
        log("[Theorist] ERROR: Part 1 not found in ledger. Cannot proceed.")
        return False

    # --- Decide whether the debate stage should activate ---
    # Gate: activate only when stagnating or declining (>= 15 trials).
    # This avoids token cost during normal improving runs.
    _results_log = thread_safe_state.get("results_log", [])
    _selected_model = thread_safe_state.get("selected_model", "unknown")
    _arch_trials = [t for t in _results_log if t.get("architecture") == _selected_model]
    _arch_trial_count = len(_arch_trials)
    _debate_active = False
    if _arch_trial_count >= 15:
        try:
            import numpy as np
            _accs = [t.get("mean_accuracy", 0.0) for t in _arch_trials[-10:]]
            _slope = float(np.polyfit(range(len(_accs)), _accs, 1)[0])
            _debate_active = abs(_slope) < 0.0005 or _slope < 0  # STAGNANT or DECLINING
        except Exception:
            _debate_active = False
    log(f"[Theorist] Hypothesis Debate stage: {'ACTIVE' if _debate_active else 'SKIPPED'} "
        f"(trials={_arch_trial_count}, gate=>=15+stagnant/declining)")

    # Define the pipeline — debate stage inserted conditionally
    stages = [
        CheckpointStage(
            name="knowledge_scout",
            marker=SECTION_MARKERS["part_2"],
            runner=lambda content, state: _run_knowledge_scout(content, state, priority_override, constraints, notes),
            max_retries=3,
        ),
        CheckpointStage(
            name="hypothesis_generation",
            marker=SECTION_MARKERS["hypothesis"],
            runner=lambda content, state: _run_hypothesis_generation(content, state, priority_override, constraints, notes),
            max_retries=3,
        ),
    ]

    if _debate_active:
        stages.append(CheckpointStage(
            name="hypothesis_debate",
            marker="**Hypothesis Debate (Critic ↔ Refiner):**",
            runner=lambda content, state: _run_hypothesis_debate(content, state),
            max_retries=2,  # One retry — if debate fails, skip it rather than halting
        ))

    stages += [
        CheckpointStage(
            name="objective_expansion",
            marker=SECTION_MARKERS["objectives"],
            runner=lambda content, state: _run_objective_expansion(content, state),
            max_retries=3,
        ),
        CheckpointStage(
            name="budgeting_assembly",
            marker=SECTION_MARKERS["part_3"],
            runner=lambda content, state: _run_budgeting_assembly(content, state),
            max_retries=3,
        ),
    ]

    for stage in stages:
        # Re-read ledger each time (previous stage may have written to it)
        ledger_content = read_ledger(ledger_path)

        if stage.marker in ledger_content:
            log(f"[Theorist] Stage '{stage.name}' already complete — skipping.")
            continue

        success = False
        for attempt in range(stage.max_retries):
            try:
                log(f"[Theorist] Running stage '{stage.name}' (attempt {attempt + 1}/{stage.max_retries})...")
                output = stage.runner(ledger_content, thread_safe_state)
                append_section(ledger_path, output)
                log(f"[Theorist] Stage '{stage.name}' completed successfully.")
                success = True
                break
            except (LLMCallError, Exception) as e:
                log(f"[Theorist] Stage '{stage.name}' attempt {attempt + 1} failed: {e}")
                if attempt < stage.max_retries - 1:
                    backoff = 2 ** attempt
                    log(f"[Theorist] Retrying in {backoff}s...")
                    time.sleep(backoff)

        if not success:
            # Debate stage is non-blocking — a failure skips it without halting the pipeline
            if stage.name == "hypothesis_debate":
                log("[Theorist] Debate stage failed — skipping gracefully, original hypotheses will be used.")
                continue
            halt_msg = (
                f"\n**[THEORIST HALTED]** Failed at stage: {stage.name}. "
                f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n"
            )
            append_section(ledger_path, halt_msg)
            log(f"[Theorist] HALTED at stage '{stage.name}'. Check ledger for details.")
            return False

    log("[Theorist] Deep Search pipeline completed successfully. Parts 2-3 written.")
    return True
