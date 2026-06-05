# summarizer_ledger.py
"""
Single Ledger Architecture — Strategic Summarizer & History Chronicle.

Two responsibilities:
1. run_multi_agent_summarizer: at cycle end, distills Part 1 + Part 4 into a
   compact Strategic Lesson saved as workspace/ledger_appendix_{cycle}.md.

2. run_history_summarizer: at the start of every new cycle (from cycle 2 onward),
   reads ALL existing appendix files and synthesizes a "Research Chronicle" (~250 words)
   to inject as Part 0. Uses a lazy threshold — activates only when >= 2 appendices exist.
"""

import json
import os
import re
from typing import Dict, Any, Optional

from ui_logger import log
from ledger_schema import parse_section, WORKSPACE_DIR
from cluster.prompts import (
    HISTORY_SUMMARIZER_PROMPT,
    HISTORY_CRITIC_PROMPT,
    HISTORY_CONSOLIDATOR_PROMPT,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_agent(thread_safe_state: Dict[str, Any]):
    agent = thread_safe_state.get("agent")
    if not agent:
        raise ValueError("Agent not found in thread_safe_state.")
    return agent


def _llm(agent, system: str, user: str) -> str:
    response, _ = agent.run_turn(system, [{"role": "user", "parts": [user]}])
    return response or ""


def _appendix_path(cycle_num: int, workspace_dir: str) -> str:
    return os.path.join(workspace_dir, f"ledger_appendix_{cycle_num}.md")


def _load_all_appendices(workspace_dir: str) -> list[tuple[int, str]]:
    """Return list of (cycle_num, content) sorted by cycle_num ascending."""
    pattern = re.compile(r"^ledger_appendix_(\d+)\.md$")
    results = []
    for fname in os.listdir(workspace_dir):
        m = pattern.match(fname)
        if m:
            cycle_num = int(m.group(1))
            path = os.path.join(workspace_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    results.append((cycle_num, content))
            except Exception as e:
                log(f"[Summarizer] Warning: could not read {fname}: {e}")
    results.sort(key=lambda x: x[0])
    return results


def _write_appendix(cycle_num: int, content: str, workspace_dir: str) -> None:
    path = _appendix_path(cycle_num, workspace_dir)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    log(f"[Summarizer] Appendix written: ledger_appendix_{cycle_num}.md")


# ---------------------------------------------------------------------------
# Phase 1: Per-cycle summarizer (called at end of each epoch)
# ---------------------------------------------------------------------------

def run_multi_agent_summarizer(
    ledger_path: str,
    thread_safe_state: Dict[str, Any],
    workspace_dir: str = WORKSPACE_DIR,
    architecture_filter: Optional[str] = None,
) -> Optional[str]:
    """
    Distill the completed cycle into a Strategic Lesson and persist it as
    workspace/ledger_appendix_{cycle}.md.

    Returns the lesson text (used by epoch_runner to pass as preamble fallback),
    or None on failure.
    """
    # Determine cycle number from ledger filename
    cycle_num = 0
    m = re.search(r"cycle_(\d+)_ledger\.md", os.path.basename(ledger_path))
    if m:
        cycle_num = int(m.group(1))

    # Skip if appendix already exists (checkpoint recovery)
    appendix_path = _appendix_path(cycle_num, workspace_dir)
    if os.path.exists(appendix_path):
        log(f"[Summarizer] Appendix for cycle {cycle_num} already exists — skipping.")
        try:
            with open(appendix_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return None

    summary_ctx = f"for architecture '{architecture_filter}'" if architecture_filter else f"for cycle {cycle_num}"
    log(f"[Summarizer] Starting per-cycle summarizer {summary_ctx}...")

    part_0 = parse_section(ledger_path, "Part 0: Strategic Continuity (Context Injection)") or "(No prior Research Chronicle — first cycle.)"
    part_1 = parse_section(ledger_path, "Part 1: Analytical Post-Mortem") or "No analytical post-mortem found."
    part_2 = parse_section(ledger_path, "Part 2: Strategic Context") or "No Theorist context found."
    part_3 = parse_section(ledger_path, "Part 3: Strategic Plan & Objectives") or "No strategic plan found."
    part_4 = parse_section(ledger_path, "Part 4: Decider's Execution Log") or "No execution log found."

    # Gather Decider context variables from state
    current_model = thread_safe_state.get("selected_model", "unknown")
    raw_data_source = thread_safe_state.get("raw_data_source", thread_safe_state.get("selected_raw_data_source", "unknown"))

    # Load the same critic_warning the Decider used this cycle (from last appendix)
    critic_warning = "(No critic warning available.)"
    try:
        import re as _re
        _appendix_files = sorted(
            [f for f in os.listdir(workspace_dir) if _re.match(r"ledger_appendix_\d+\.md", f)],
            key=lambda x: int(_re.search(r"(\d+)", x).group(1))
        )
        # Use the appendix BEFORE the current cycle (i.e. last one written, not the one being written now)
        _prior = [f for f in _appendix_files if int(_re.search(r"(\d+)", f).group(1)) < cycle_num]
        if _prior:
            with open(os.path.join(workspace_dir, _prior[-1]), "r", encoding="utf-8") as _f:
                _ac = _f.read()
            _cm = _re.search(r"\*\*Critique:\*\*(.*?)(?=\n#|\Z)", _ac, _re.DOTALL)
            if _cm:
                critic_warning = _cm.group(1).strip()[:400]
    except Exception as _e:
        log(f"[Summarizer] Warning: could not load critic_warning: {_e}")

    try:
        agent = _get_agent(thread_safe_state)

        # 1. Summarizer — extract factual lesson
        log("[Summarizer] Step 1/3: Summarizer...")
        raw_summary = _llm(
            agent,
            "You are a precise data analyst. Be factual and concise.",
            HISTORY_SUMMARIZER_PROMPT.format(
                cycle_num=cycle_num,
                current_model=current_model,
                raw_data_source=raw_data_source,
                critic_warning=critic_warning,
                part_0_content=part_0,
                part_1_content=part_1,
                part_2_content=part_2,
                part_3_content=part_3,
                part_4_content=part_4,
            ),
        )

        # 2. Critic — challenge the summary
        log("[Summarizer] Step 2/3: Critic...")
        critique = _llm(
            agent,
            "You are a critical scientific reviewer. Be direct.",
            HISTORY_CRITIC_PROMPT.format(cycle_summary=raw_summary),
        )

        # 3. Consolidator — merge into final lesson
        log("[Summarizer] Step 3/3: Consolidator...")
        lesson = _llm(
            agent,
            "You are a strategic planner. Be concise and actionable.",
            HISTORY_CONSOLIDATOR_PROMPT.format(
                all_lessons=f"**Cycle {cycle_num} Summary:**\n{raw_summary}\n\n**Critique:**\n{critique}"
            ),
        )

        # Persist as appendix — keep Critique as a labeled section so decider_ledger can extract it
        appendix_content = (
            f"# Ledger Appendix — Cycle {cycle_num}\n\n"
            f"## Strategic Lesson\n\n{lesson.strip()}\n\n"
            f"## Critique\n\n**Critique:** {critique.strip()}\n"
        )
        _write_appendix(cycle_num, appendix_content, workspace_dir)

        # Also persist lesson in long-term vector memory so it's
        # retrievable by recall_relevant_memories in future Theorist cycles.
        try:
            from tools import memorize_finding
            memorize_finding(
                finding=lesson.strip(),
                experiment_name=f"Cycle_{cycle_num}_Strategic_Lesson",
                thread_safe_state=thread_safe_state,
                category="strategic_lesson",
            )
            log(f"[Summarizer] Lesson for cycle {cycle_num} saved to long-term memory.")
        except Exception as me:
            log(f"[Summarizer] WARNING: Could not save lesson to memory: {me}")

        return lesson.strip()

    except Exception as e:
        log(f"[Summarizer] ERROR: Per-cycle summarizer failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Phase 2: History synthesizer (called at start of new cycle, from cycle 2+)
# ---------------------------------------------------------------------------

LAZY_THRESHOLD = 2  # Activate only when >= this many appendices exist


def run_history_summarizer(
    thread_safe_state: Dict[str, Any],
    workspace_dir: str = WORKSPACE_DIR,
) -> Optional[str]:
    """
    Read all ledger_appendix_*.md files and synthesize a Research Chronicle
    (~250 words) to inject as Part 0 of the current cycle.

    Returns None if fewer than LAZY_THRESHOLD appendices exist (not enough history).
    """
    appendices = _load_all_appendices(workspace_dir)

    if len(appendices) < LAZY_THRESHOLD:
        log(f"[HistorySummarizer] Only {len(appendices)} appendix(es) found — "
            f"lazy threshold ({LAZY_THRESHOLD}) not reached. Skipping chronicle.")
        return None

    log(f"[HistorySummarizer] Synthesizing Research Chronicle from {len(appendices)} appendices...")

    # Build combined input — label each cycle clearly
    all_lessons = "\n\n".join(
        f"### Cycle {cycle_num}\n{content}" for cycle_num, content in appendices
    )

    try:
        agent = _get_agent(thread_safe_state)

        chronicle = _llm(
            agent,
            "You are a strategic research historian. Be concise and decisive.",
            HISTORY_CONSOLIDATOR_PROMPT.format(all_lessons=all_lessons),
        )

        if not chronicle.strip():
            log("[HistorySummarizer] LLM returned empty chronicle.")
            return None

        log(f"[HistorySummarizer] Chronicle synthesized ({len(chronicle.split())} words).")
        return chronicle.strip()

    except Exception as e:
        log(f"[HistorySummarizer] ERROR: History synthesis failed: {e}")
        return None


# ---------------------------------------------------------------------------
# On-demand Research Chronicle — technical report (deterministic table + LLM digest)
# Pull-based: generated at session start and on Theorist request (refresh_chronicle tool),
# written to a standalone file (NOT the ledger). See tools.refresh_chronicle.
# ---------------------------------------------------------------------------

CHRONICLE_FILE = "research_chronicle.md"


def _build_technical_table(thread_safe_state: Dict[str, Any], workspace_dir: str) -> str:
    """Deterministic technical digest (no LLM): per-architecture metrics from results_log
    + a cycle/ledger index with timestamps. Exact, cheap, reproducible."""
    from collections import defaultdict

    results_log = thread_safe_state.get("results_log", []) or []
    by_arch = defaultdict(list)
    for t in results_log:
        by_arch[t.get("architecture", "unknown")].append(t)

    def _vals(trials, key):
        return [t.get(key) for t in trials if t.get(key) is not None]

    arch_rows = []
    for arch, trials in by_arch.items():
        hos = _vals(trials, "held_out_test_accuracy")
        shs = _vals(trials, "shifted_test_accuracy")
        gaps = _vals(trials, "mean_overfitting_score")
        peak = max(hos) if hos else 0.0
        arch_rows.append((
            peak, arch, len(trials),
            f"{peak:.4f}" if hos else "—",
            f"{sum(hos)/len(hos):.4f}" if hos else "—",
            f"{max(shs):.4f}" if shs else "—",
            f"{min(gaps):.4f}" if gaps else "—",
        ))
    arch_rows.sort(key=lambda r: -r[0])
    arch_table = ["| Architecture | n | peak HO | mean HO | best shifted | min gap |",
                  "|---|---|---|---|---|---|"]
    for _, arch, n, peak, mean_ho, best_sh, min_gap in arch_rows:
        arch_table.append(f"| {arch} | {n} | {peak} | {mean_ho} | {best_sh} | {min_gap} |")

    # Cycle / ledger index with timestamps (mtime)
    from datetime import datetime as _dt
    ledger_re = re.compile(r"^cycle_(\d+)_ledger\.md$")
    cyc_rows = []
    try:
        for fname in os.listdir(workspace_dir):
            m = ledger_re.match(fname)
            if m:
                cyc = int(m.group(1))
                p = os.path.join(workspace_dir, fname)
                ts = _dt.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")
                has_app = os.path.exists(_appendix_path(cyc, workspace_dir))
                cyc_rows.append((cyc, fname, ts, has_app))
        cyc_rows.sort(key=lambda r: r[0])
    except Exception as e:
        log(f"[Chronicle] Warning: could not index ledgers: {e}")
    cyc_table = ["| Cycle | Ledger | Timestamp | Appendix |", "|---|---|---|---|"]
    for cyc, fname, ts, has_app in cyc_rows:
        cyc_table.append(f"| {cyc} | {fname} | {ts} | {'yes' if has_app else 'no'} |")

    # Finetune runs (domain adaptation) — richer technical fields from finetune_results_log.json
    def _f(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "—"

    ft_rows = []
    try:
        ft_path = None
        try:
            from state_manager import PERSISTENT_PATHS as _PP
            ft_path = _PP.get("finetune_results_log")
        except Exception:
            pass
        if not ft_path:
            ft_path = os.path.join("processed_data", "logs", "finetune_results_log.json")
        if os.path.exists(ft_path):
            with open(ft_path, "r", encoding="utf-8") as f:
                ft_data = json.load(f)
            for entry in (ft_data or []):
                flat = entry.get("flat", entry) if isinstance(entry, dict) else {}
                pcs = flat.get("per_class_shifted") or {}
                worst = "—"
                if pcs:
                    wc, wv = min(((c, (d or {}).get("acc", 0.0)) for c, d in pcs.items()), key=lambda x: x[1])
                    worst = f"{wc}:{wv:.2f}"
                delta = flat.get("delta_pretrain_to_finetune")
                ft_rows.append(
                    f"| {flat.get('architecture', '?')} | {flat.get('finetune_unfreeze_last_n', '?')} "
                    f"| {_f(flat.get('pretrain_ho_accuracy'))} | {_f(flat.get('held_out_test_accuracy'))} "
                    f"| {(f'{delta:+.4f}' if isinstance(delta, (int, float)) else '—')} "
                    f"| {_f(flat.get('shifted_test_accuracy'))} | {_f(flat.get('finetune_train_accuracy'))} | {worst} |"
                )
    except Exception as e:
        log(f"[Chronicle] Warning: could not read finetune_results_log: {e}")

    ft_table = ""
    if ft_rows:
        ft_table = (
            "\n\n## Finetune runs (domain adaptation)\n"
            "| Arch | unfreeze | pretrain HO | ft HO | delta | shifted | final_train_acc | worst-class |\n"
            "|---|---|---|---|---|---|---|---|\n" + "\n".join(ft_rows)
        )

    return (
        f"## Per-architecture metrics (from {len(results_log)} trials)\n"
        + "\n".join(arch_table)
        + "\n\n## Cycle / ledger index\n"
        + "\n".join(cyc_table)
        + ft_table
    )


def build_research_chronicle(thread_safe_state: Dict[str, Any], workspace_dir: str = WORKSPACE_DIR) -> Optional[str]:
    """Build the full technical Research Chronicle (deterministic table + LLM-condensed
    insights) and write it to workspace/research_chronicle.md. Returns the text, or None
    if there is no history yet."""
    from datetime import datetime as _dt

    table = _build_technical_table(thread_safe_state, workspace_dir)
    appendices = _load_all_appendices(workspace_dir)

    insights = "(no history yet)"
    if appendices:
        all_lessons = "\n\n".join(f"### Cycle {n}\n{c}" for n, c in appendices)
        try:
            agent = _get_agent(thread_safe_state)
            insights = _llm(
                agent,
                "You are a technical research historian. Output a TERSE technical digest, "
                "NOT a narrative. Use only these sections as bullet lists citing cycle numbers: "
                "DEAD-ENDS (arch/representation + reason), UNRESOLVED BUGS (cycle + error), "
                "BEST STABLE CONFIGS (arch + key params + metric). No prose.",
                f"Per-cycle strategic lessons:\n\n{all_lessons}",
            ).strip() or "(condensation returned empty)"
        except Exception as e:
            insights = f"(condensation unavailable: {e})"

    n_cycles = len(appendices)
    ts = _dt.now().strftime("%Y-%m-%d %H:%M")
    chronicle = (
        f"# Research Chronicle (technical) — updated {ts} — {n_cycles} cycles\n\n"
        f"{table}\n\n## Condensed insights\n{insights}\n"
    )
    try:
        path = os.path.join(workspace_dir, CHRONICLE_FILE)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(chronicle)
        os.replace(tmp, path)
        log(f"[Chronicle] Written {CHRONICLE_FILE} ({len(chronicle.split())} words, {n_cycles} cycles).")
    except Exception as e:
        log(f"[Chronicle] ERROR writing {CHRONICLE_FILE}: {e}")
    return chronicle
