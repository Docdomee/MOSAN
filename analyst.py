# analyst.py
"""
Single Ledger Architecture — Phase 1: Lead Analyst Agent.

The Lead Analyst is a post-execution agent that reads raw data from
state_manager, MLflow, and graph memory, then writes Part 1 (Analytical
Post-Mortem) of the next cycle's ledger.

Part 1 has two sections:
  A) QUANTITATIVE DECISION BRIEFING — deterministic, built by `build_decision_briefing`
     (Performance & Gap, statistical analysis, cross-arch champions, Opportunity Map).
     Written verbatim; the LLM never re-types these numbers.
  B) INTERPRETATION & FLAGS — thin LLM layer (deltas, anomalies, stagnation, objective
     evaluation) on top of the briefing.
"""

import json
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import mlflow

from ledger_schema import (
    append_section,
    parse_section,
    read_ledger,
    SECTION_MARKERS,
)
from state_manager import PERSISTENT_PATHS
from ui_logger import log
from cluster.prompts import LEDGER_ANALYST_PROMPT


def _gather_analysis_data(thread_safe_state: Dict[str, Any]) -> str:
    """
    Actively call analysis tools to build a data-derived picture of what happened.

    Returns the statistical-analysis body (A.2) of the deterministic Decision
    Briefing — correlations, best/worst, importance, recent trials, tradeoffs,
    top-5, optimization trend — written verbatim into Part 1 (no LLM compression).

    Uses tools directly (read-only, no graph memory updates needed).
    """
    selected_model = thread_safe_state.get("selected_model")
    arch_filter = selected_model if selected_model else None

    sections = []

    # 1. Correlation matrix — which hyperparameters drove accuracy
    try:
        from tools import get_correlation_matrix
        result = get_correlation_matrix(
            thread_safe_state=thread_safe_state,
            architecture_filter=arch_filter,
        )
        if result.get("status") == "completed":
            corr = result.get("correlation_analysis", {}).get("mean_accuracy", {})
            if corr:
                lines = ["Hyperparameter Correlations with Accuracy (|r| > 0.2 shown):"]
                for param, stats in list(corr.items())[:8]:
                    r = stats.get("correlation", 0)
                    if abs(r) > 0.2:
                        sig = ""
                        if stats.get("significant") is True:
                            sig = " [significant]"
                        elif stats.get("significant") is False:
                            sig = " [not significant]"
                        lines.append(f"  - {param}: r={r:.3f}{sig}")
                if len(lines) > 1:
                    sections.append("\n".join(lines))
                else:
                    sections.append("Hyperparameter Correlations: No strong correlations found (|r| < 0.2).")
        elif result.get("message"):
            sections.append(f"Correlation Matrix: {result['message']}")
    except Exception as e:
        sections.append(f"Correlation Matrix: Error ({e}).")

    # 2. Best vs worst trial comparison
    try:
        from tools import analyze_best_vs_worst_trials
        result = analyze_best_vs_worst_trials(
            thread_safe_state=thread_safe_state,
            architecture_filter=arch_filter,
        )
        if result.get("status") == "completed":
            best = result.get("best_trials_stats", {})
            worst = result.get("worst_trials_stats", {})
            overfitting = result.get("overfitting_check", "")
            lines = ["Best vs Worst Trial Comparison (top/bottom 25%):"]
            # Key parameters to compare
            key_params = ["learning_rate", "dropout_rate", "weight_decay", "batch_size",
                          "kernel_size", "num_filters", "mean_accuracy"]
            for p in key_params:
                b = best.get(p)
                w = worst.get(p)
                if b is not None and w is not None:
                    try:
                        lines.append(f"  - {p}: best={b:.4f}, worst={w:.4f}")
                    except (TypeError, ValueError):
                        lines.append(f"  - {p}: best={b}, worst={w}")
            if overfitting:
                lines.append(f"  Overfitting Check: {overfitting}")
            if len(lines) > 1:
                sections.append("\n".join(lines))
        elif result.get("message"):
            sections.append(f"Best/Worst Analysis: {result['message']}")
    except Exception as e:
        sections.append(f"Best/Worst Analysis: Error ({e}).")

    # 3. Parameter importance (Random Forest)
    try:
        from tools import get_parameter_importance
        result = get_parameter_importance(
            thread_safe_state=thread_safe_state,
            architecture_filter=arch_filter,
        )
        if result.get("status") == "completed":
            importance = result.get("importance", {})
            if importance:
                top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                lines = ["Parameter Importance (Random Forest, top 5):"]
                for param, score in top:
                    lines.append(f"  - {param}: {score:.3f}")
                sections.append("\n".join(lines))
        elif result.get("message"):
            sections.append(f"Parameter Importance: {result['message']}")
    except Exception as e:
        sections.append(f"Parameter Importance: Error ({e}).")

    # 4. Optimization status (plateau / flag detection)
    try:
        from tools import get_optimization_status
        opt_target = thread_safe_state.get("optimization_target", 0.95)
        result = get_optimization_status(
            thread_safe_state=thread_safe_state,
            optimization_threshold=opt_target
        )
        if result.get("status") == "completed":
            status_lines = []
            models = result.get("architectures", {})
            if models:
                status_lines.append("Architecture Optimization Status:")
                for arch, info in models.items():
                    opt_status = info.get("status", "unknown")
                    trials = info.get("trial_count", 0)
                    best = info.get("best_accuracy", 0.0)
                    status_lines.append(
                        f"  - {arch}: {opt_status} ({trials} trials, best={best:.4f})"
                    )
            if status_lines:
                sections.append("\n".join(status_lines))
    except Exception as e:
        sections.append(f"Optimization Status: Error ({e}).")

    # 5. Recent trials (last 5, full params) — raw evidence for delta analysis
    try:
        from tools import get_recent_trials
        result = get_recent_trials(
            thread_safe_state=thread_safe_state,
            n=5,
            architecture_filter=arch_filter,
        )
        if result.get("status") == "completed":
            trials = result.get("recent_trials", [])
            if trials:
                lines = ["Recent Trials (last 5, current architecture):"]
                for t in trials:
                    acc = t.get("mean_accuracy") or 0.0
                    train_acc = t.get("mean_train_accuracy") or 0.0
                    gap = train_acc - acc if train_acc else 0.0
                    key_params = {k: v for k, v in t.items()
                                  if k in ["learning_rate", "dropout_rate", "batch_size",
                                           "kernel_size", "num_filters", "num_layers",
                                           "weight_decay", "optimizer"]}
                    lines.append(f"  - Test={acc:.4f}, Train={train_acc:.4f}, Gap={gap:.4f} | {key_params}")
                sections.append("\n".join(lines))
    except Exception as e:
        sections.append(f"Recent Trials: Error ({e}).")

    # 6. Hyperparameter tradeoffs — which param values consistently win vs lose
    try:
        from tools import analyze_hyperparameter_tradeoffs
        result = analyze_hyperparameter_tradeoffs(
            thread_safe_state=thread_safe_state,
            architecture_filter=arch_filter,
        )
        if result.get("status") == "completed":
            tradeoffs = result.get("tradeoffs", {})
            if tradeoffs:
                lines = ["Hyperparameter Tradeoffs (best vs worst value per param):"]
                for param, data in list(tradeoffs.items())[:6]:
                    analysis = data.get("analysis", [])
                    if not analysis:
                        continue
                    # Sort by mean_accuracy descending, show best and worst value
                    try:
                        sorted_vals = sorted(analysis, key=lambda x: x.get("mean_accuracy", 0), reverse=True)
                        best_val = sorted_vals[0]
                        worst_val = sorted_vals[-1]
                        lines.append(
                            f"  - {param}: best={best_val.get(param, '?')} "
                            f"(acc={best_val.get('mean_accuracy', 0):.4f}, n={best_val.get('trial_count', 0)}), "
                            f"worst={worst_val.get(param, '?')} "
                            f"(acc={worst_val.get('mean_accuracy', 0):.4f}, n={worst_val.get('trial_count', 0)})"
                        )
                    except Exception:
                        continue
                if len(lines) > 1:
                    sections.append("\n".join(lines))
    except Exception as e:
        sections.append(f"Hyperparameter Tradeoffs: Error ({e}).")

    # 7. Top-5 trials of all time (absolute best, not just recent)
    try:
        from tools import get_sorted_trials
        result = get_sorted_trials(
            thread_safe_state=thread_safe_state,
            sort_by="mean_accuracy",
            ascending=False,
            n=5,
            architecture_filter=arch_filter,
        )
        key = [k for k in result if k.startswith("top_")]
        if result.get("status") == "completed" and key:
            trials = result[key[0]]
            if trials:
                lines = ["Top-5 Trials of All Time (by accuracy):"]
                for i, t in enumerate(trials, 1):
                    acc = t.get("mean_accuracy") or 0.0
                    train_acc = t.get("mean_train_accuracy") or 0.0
                    gap = train_acc - acc if train_acc else 0.0
                    lr = t.get("learning_rate", "?")
                    dr = t.get("dropout_rate", "?")
                    ks = t.get("kernel_size", "?")
                    lines.append(
                        f"  #{i}: acc={acc:.4f}, gap={gap:.4f} | "
                        f"lr={lr}, dropout={dr}, kernel={ks}"
                    )
                sections.append("\n".join(lines))
        elif result.get("message"):
            sections.append(f"Top-5 Trials: {result['message']}")
    except Exception as e:
        sections.append(f"Top-5 Trials: Error ({e}).")

    # 8. Optimization trend — slope of accuracy over trial sequence (stagnation detection)
    try:
        results_log = thread_safe_state.get("results_log", [])
        arch_trials = [
            t for t in results_log
            if arch_filter is None or t.get("architecture") == arch_filter
        ]
        if len(arch_trials) >= 5:
            import numpy as np
            accs = [(t.get("mean_accuracy") or 0.0) for t in arch_trials]
            x = np.arange(len(accs), dtype=float)
            slope = float(np.polyfit(x, accs, 1)[0])
            # Classify stagnation
            if abs(slope) < 0.0005:
                status_label = "STAGNANT"
            elif slope > 0:
                status_label = "IMPROVING"
            else:
                status_label = "DECLINING"
            sections.append(
                f"Optimization Trend ({arch_filter or 'global'}, {len(accs)} trials): "
                f"slope={slope:.5f} -> {status_label}"
            )
        else:
            sections.append(
                f"Optimization Trend: Not enough trials yet "
                f"({len(arch_trials)} trials, need >= 5)."
            )
    except Exception as e:
        sections.append(f"Optimization Trend: Error ({e}).")

    if not sections:
        return "Analysis Tools: No data available yet (likely too few trials)."

    return "\n\n".join(sections)


def _gather_cross_architecture_summary(thread_safe_state: Dict[str, Any], min_trials: int = 3) -> str:
    """
    Compact champion list across all architectures:
    - Best HO: highest held_out_test_accuracy
    - Best Shifted: highest shifted_test_accuracy
    - Best Gap: smallest (HO - Shifted) among entries with both metrics

    Only architectures with >= min_trials trials are considered.
    Returns empty string if fewer than 2 distinct architectures exist.
    """
    results_log = thread_safe_state.get("results_log", [])
    if not results_log:
        return ""

    # Group by architecture
    by_arch: Dict[str, List[dict]] = {}
    for t in results_log:
        arch = t.get("architecture") or t.get("manifest_name", "unknown")
        by_arch.setdefault(arch, []).append(t)

    # Filter by min_trials
    qualified = {a: trials for a, trials in by_arch.items() if len(trials) >= min_trials}
    if len(qualified) < 2:
        return ""

    # Champions
    best_ho_arch, best_ho_val, best_ho_trial = None, -1.0, None
    best_sh_arch, best_sh_val, best_sh_trial = None, -1.0, None
    best_gap_arch, best_gap_val, best_gap_trial = None, float("inf"), None

    for arch, trials in qualified.items():
        for t in trials:
            ho = t.get("held_out_test_accuracy")
            sh = t.get("shifted_test_accuracy")

            if ho is not None and ho > best_ho_val:
                best_ho_val, best_ho_arch, best_ho_trial = ho, arch, t

            if sh is not None and sh > best_sh_val:
                best_sh_val, best_sh_arch, best_sh_trial = sh, arch, t

            if ho is not None and sh is not None:
                gap = ho - sh
                if gap < best_gap_val:
                    best_gap_val, best_gap_arch, best_gap_trial = gap, arch, t

    def _trial_note(t: dict) -> str:
        if t is None:
            return ""
        params = t.get("params", {})
        snippets = []
        for k in ["learning_rate", "num_conv_layers", "dropout_rate"]:
            if k in params:
                snippets.append(f"{k}={params[k]}")
        return f"  params: {', '.join(snippets)}" if snippets else ""

    # Separate finetune trials from regular trials for Shifted champion
    finetune_trials_all = [t for trials in qualified.values() for t in trials
                           if t.get("source") == "finetune_trial"]
    best_sh_finetune = max(
        (t for t in finetune_trials_all if t.get("shifted_test_accuracy") is not None),
        key=lambda t: t["shifted_test_accuracy"],
        default=None,
    )

    lines = ["## Cross-Architecture Champions (architectures with ≥{} trials only)".format(min_trials)]
    lines.append("Shifted = TRUE deployment holdout (cross-session). HO = intra-session only, used for HP search.")
    lines.append("")

    if best_ho_arch:
        n = len(qualified[best_ho_arch])
        note = _trial_note(best_ho_trial)
        lines.append(f"🏆 Best HO:      `{best_ho_arch}` — HO={best_ho_val:.4f}  ({n} trials){note}")

    # Shifted champion: prefer finetune_trial source; fall back to normal trials with explicit warning
    if best_sh_finetune is not None:
        ft_arch = best_sh_finetune.get("architecture") or best_sh_finetune.get("manifest_name", "?")
        ft_sh = best_sh_finetune["shifted_test_accuracy"]
        ft_ho = best_sh_finetune.get("held_out_test_accuracy")
        ft_delta = best_sh_finetune.get("delta_pretrain_to_finetune")
        delta_str = f"  delta={ft_delta:+.4f}" if ft_delta is not None else ""
        ho_str = f"  HO={ft_ho:.4f}" if ft_ho is not None else ""
        ft_fta = best_sh_finetune.get("finetune_train_accuracy")
        fta_str = f"  final_train_acc={ft_fta:.4f}" if ft_fta is not None else ""
        ft_unf = best_sh_finetune.get("finetune_unfreeze_last_n")
        unf_str = f"  ft_unfreeze_last_n={ft_unf}" if ft_unf is not None else ""
        pcs = best_sh_finetune.get("per_class_shifted") or {}
        worst_str = ""
        if pcs:
            worst_cls, worst_acc = min(
                ((c, d.get("acc", 0.0)) for c, d in pcs.items()), key=lambda x: x[1]
            )
            worst_str = f"  worst-class={worst_cls}:{worst_acc:.2f}"
        lines.append(
            f"🎯 Best Shifted: `{ft_arch}` — Shifted={ft_sh:.4f}{ho_str}{delta_str}{fta_str}{unf_str}{worst_str}  [finetune_trial ✓]"
        )
    elif best_sh_arch and best_sh_val > 0:
        n = len(qualified[best_sh_arch])
        note = _trial_note(best_sh_trial)
        lines.append(
            f"🎯 Best Shifted: `{best_sh_arch}` — Shifted={best_sh_val:.4f}  ({n} trials){note}"
            f"  ⚠️ SOURCE: normal trial (no finetune yet) — run `run_training_trial_with_finetune` to improve"
        )
    else:
        lines.append("🎯 Best Shifted: no results yet — run `run_training_trial_with_finetune`")

    if best_gap_arch and best_gap_val < float("inf"):
        n = len(qualified[best_gap_arch])
        note = _trial_note(best_gap_trial)
        lines.append(f"⚖️  Best Gap:     `{best_gap_arch}` — Gap={best_gap_val:.4f}  ({n} trials){note}")
    else:
        lines.append("⚖️  Best Gap:     not yet computable (need finetune results)")

    return "\n".join(lines)


def _gather_state_summary(thread_safe_state: Dict[str, Any]) -> str:
    """
    Extract current performance data from the session state.
    Returns a structured text block for the Analyst LLM.
    """
    results_log = thread_safe_state.get("results_log", [])
    best_acc = thread_safe_state.get("best_accuracy", 0.0)
    selected_model = thread_safe_state.get("selected_model", "Unknown")
    global_step = thread_safe_state.get("global_step_counter", 0)

    # Compute true best held-out accuracy from results_log
    results_log_tmp = thread_safe_state.get("results_log", [])
    best_heldout_ever = max((t.get("held_out_test_accuracy") or 0.0 for t in results_log_tmp), default=0.0)

    lines = []
    lines.append(f"Global Step: {global_step}")
    lines.append(f"Selected Model: {selected_model}")
    lines.append(f"Global Best Held-Out Accuracy (TRUE): {best_heldout_ever:.4f}")
    lines.append(f"Global Best Val CV Accuracy (internal, DO NOT USE as primary metric): {best_acc:.4f}")

    # Current architecture trials
    arch_trials = [t for t in results_log if t.get("architecture") == selected_model]
    if arch_trials:
        arch_best_heldout = max((t.get("held_out_test_accuracy") or 0.0) for t in arch_trials)
        arch_best_val = max((t.get("mean_accuracy") or 0.0) for t in arch_trials)
        lines.append(f"Current Architecture ({selected_model}) Best Held-Out (TRUE): {arch_best_heldout:.4f}")
        lines.append(f"Current Architecture ({selected_model}) Best Val CV (internal): {arch_best_val:.4f}")
        lines.append(f"Total Trials for {selected_model}: {len(arch_trials)}")
    else:
        lines.append(f"No trials completed for {selected_model} yet.")

    # Last 10 trials (architecture-agnostic)
    if results_log:
        last_trials = results_log[-10:]
        lines.append(f"\nLast {len(last_trials)} Trials:")
        for t in last_trials:
            m_name = t.get("manifest_name", "N/A")
            arch = t.get("architecture", "Unknown")
            params = t.get("params", {})

            # Finetune trials carry different fields: mean_accuracy / mean_train_accuracy
            # are null, so the generic line below would print Train=0/Gap=0 (misleading).
            # Surface instead the finetune-specific signals the freeze-escalation heuristic
            # depends on: final_train_acc (low ⇒ frozen backbone can't fit the calibration
            # set), per-class shifted (collapse detector), and the unfreeze depth used.
            if t.get("source") == "finetune_trial":
                ft_ho = t.get("held_out_test_accuracy")
                shifted = t.get("shifted_test_accuracy")
                ftacc = t.get("finetune_train_accuracy")
                delta = t.get("delta_pretrain_to_finetune")
                unfreeze = t.get("finetune_unfreeze_last_n")
                ft_ho_s = f"{ft_ho:.4f}" if ft_ho is not None else "N/A"
                sh_s = f"{shifted:.4f}" if shifted is not None else "N/A"
                fta_s = f"{ftacc:.4f}" if ftacc is not None else "N/A"
                dl_s = f"{delta:+.4f}" if delta is not None else "N/A"
                unf_s = str(unfreeze) if unfreeze is not None else "?"
                pcs = t.get("per_class_shifted") or {}
                pc_s = ""
                if pcs:
                    pc_s = " | shifted/class: " + ", ".join(
                        f"{cls}={d.get('acc', 0.0):.2f}" for cls, d in pcs.items()
                    )
                lines.append(
                    f"  - [FINETUNE {arch}] {m_name}: finetune_HO={ft_ho_s}, Shifted={sh_s}, "
                    f"final_train_acc={fta_s} (low -> frozen backbone can't fit calibration set; "
                    f"raise ft_unfreeze_last_n), delta={dl_s}, ft_unfreeze_last_n={unf_s}{pc_s}"
                )
                continue

            acc = t.get("mean_accuracy") or 0.0
            train_acc = t.get("mean_train_accuracy") or 0.0
            overfitting = train_acc - acc if train_acc > 0 else 0.0

            held_out = t.get("held_out_test_accuracy") or 0.0
            shifted = t.get("shifted_test_accuracy")
            shifted_str = f", Shifted={shifted:.4f}" if shifted is not None else ""
            trial_line = f"  - [{arch}] {m_name}: HeldOut={held_out:.4f}{shifted_str}, ValCV={acc:.4f}, Train={train_acc:.4f}, Gap={overfitting:.4f}"

            # Include key hyperparameters if available
            key_params = {}
            for k in ["learning_rate", "dropout_rate", "weight_decay", "batch_size", "epochs", "optimizer"]:
                if k in params:
                    key_params[k] = params[k]
            if key_params:
                trial_line += f" | Params: {json.dumps(key_params)}"

            lines.append(trial_line)
    else:
        lines.append("\nNo trials completed (Fresh Start).")

    return "\n".join(lines)


def _gather_mlflow_summary(experiment_name: str = "Agent_Brain", max_runs: int = 10) -> str:
    """
    Query MLflow for recent runs and extract key metrics.
    Returns a structured text block, or a fallback message if unavailable.
    """
    try:
        experiments = mlflow.search_experiments()
        target_exp = None
        for exp in experiments:
            if exp.name == experiment_name:
                target_exp = exp
                break

        if not target_exp:
            return "MLflow: No experiment found. Skipping."

        runs = mlflow.search_runs(
            experiment_ids=[target_exp.experiment_id],
            max_results=max_runs,
            order_by=["start_time DESC"],
        )

        if runs.empty:
            return "MLflow: No runs found."

        lines = [f"MLflow Recent Runs (Experiment: {experiment_name}):"]
        for _, row in runs.iterrows():
            run_name = row.get("tags.mlflow.runName", "unnamed")
            status = row.get("status", "unknown")
            # Collect numeric metric columns
            metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
            metrics_str = ", ".join(
                f"{c.replace('metrics.', '')}={row[c]:.4f}"
                for c in metric_cols
                if not (row[c] != row[c])  # skip NaN
            )
            lines.append(f"  - {run_name} ({status}): {metrics_str if metrics_str else 'no metrics'}")

        return "\n".join(lines)

    except Exception as e:
        return f"MLflow: Query failed ({e}). Skipping."


def _gather_graph_summary(thread_safe_state: Dict[str, Any]) -> str:
    """
    Get a summary of the strategic graph state.
    Returns a text block with node/edge counts and high-value states.
    """
    try:
        graph_memory = thread_safe_state.get("strategic_graph_memory")
        if graph_memory is None:
            return "Strategic Graph: Not initialized."

        graph = graph_memory.graph
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        lines = [f"Strategic Graph: {num_nodes} nodes, {num_edges} edges."]

        # High-value states
        high_value = graph_memory.get_high_value_states(top_k=3, min_q_value=0.1)
        if high_value:
            lines.append(f"High-Value States: {', '.join(high_value)}")
        else:
            lines.append("High-Value States: None identified yet.")

        return "\n".join(lines)

    except Exception as e:
        return f"Strategic Graph: Error reading ({e})."


def _gather_previous_cycle_part1(prev_ledger_path: Optional[str]) -> str:
    """
    Read Part 1 from the previous cycle's ledger for delta comparison.
    """
    if not prev_ledger_path:
        return "Previous Cycle: None (this is the first cycle)."

    part1 = parse_section(prev_ledger_path, "Part 1: Analytical Post-Mortem")
    if part1:
        return f"Previous Cycle Analysis:\n{part1}"
    else:
        return "Previous Cycle: Ledger exists but Part 1 not found."


def _gather_previous_cycle_objectives_and_results(prev_ledger_path: Optional[str]) -> str:
    """
    Read Part 3 (Theorist objectives) and Part 4 (Decider execution log)
    from the previous cycle's ledger so the Analyst can evaluate whether
    the Decider achieved what the Theorist planned.
    """
    if not prev_ledger_path:
        return "Previous Objectives & Results: None (first cycle)."

    sections = []

    part3 = parse_section(prev_ledger_path, "Part 3: Strategic Plan & Objectives")
    if part3:
        # Prevent exact markers from triggering Theorist checkpoints in the new cycle
        part3 = part3.replace("**Hypothesis:**", "*Previous Hypothesis:*")
        part3 = part3.replace("**Semantic Objectives:**", "*Previous Objectives:*")
        part3 = part3.replace("**Step Budget:**", "*Previous Budget:*")
        
        # Keep just the objectives and hypothesis — truncate to avoid token bloat
        lines = part3.strip().splitlines()
        if len(lines) > 30:
            lines = lines[:30] + ["...(truncated)..."]
        sections.append(f"THEORIST'S OBJECTIVES (what was planned):\n" + "\n".join(lines))
    else:
        sections.append("THEORIST'S OBJECTIVES: Not found in previous ledger.")

    part4 = parse_section(prev_ledger_path, "Part 4: Decider's Execution Log")
    if part4:
        lines = part4.strip().splitlines()
        if len(lines) > 100:
            lines = lines[:100] + ["...(truncated)..."]
        sections.append(f"DECIDER'S EXECUTION LOG (what actually happened):\n" + "\n".join(lines))
    else:
        sections.append("DECIDER'S EXECUTION LOG: Not found (Decider may not have run).")

    return "\n\n".join(sections)


def _gather_current_cycle_part0(ledger_path: str) -> str:
    """
    Read Part 0 from the current cycle's ledger (injected by Epoch Runner).
    This serves as the overarching strategic directive for the new session.
    """
    part0 = parse_section(ledger_path, "Part 0: Strategic Continuity (Context Injection)")
    if part0:
        return f"Strategic Continuity Directive (Follow This Strictly):\n{part0}"
    else:
        return "Strategic Continuity Directive: None provided."


# ---------------------------------------------------------------------------
# A3: Deterministic "Quantitative Decision Briefing" (Part 1, section A)
# ---------------------------------------------------------------------------

# Tunable hyperparameters tracked by the Opportunity Map (current architecture).
_TUNABLE_PARAMS = [
    "learning_rate", "dropout_rate", "weight_decay", "batch_size",
    "kernel_size", "filters", "num_conv_layers", "dense_units", "batch_norm",
]


def _trial_params(trial: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the hyperparameter dict from a trial, tolerant of storage layout.

    Trials store HPs under `params` and/or `effective_hyperparameters`; prefer
    `params`, then effective, then the top-level dict itself.
    """
    if not isinstance(trial, dict):
        return {}
    params = trial.get("params")
    if isinstance(params, dict) and params:
        return params
    eff = trial.get("effective_hyperparameters")
    if isinstance(eff, dict) and eff:
        return eff
    return trial


def _compute_opportunity_map(thread_safe_state: Dict[str, Any], min_trials: int = 3) -> str:
    """Deterministic 'what has NOT been tried / where is the margin' map.

    Pure formatting from results_log + config. No LLM. This is the actionable
    signal the current Analyst never surfaces: it lets the Theorist pick a
    qualitatively different move instead of re-running near-duplicate configs.
    Degrades gracefully on short/empty logs.
    """
    results_log = thread_safe_state.get("results_log", []) or []
    selected_model = thread_safe_state.get("selected_model", "Unknown")
    arch_trials = [
        t for t in results_log
        if t.get("architecture") == selected_model and t.get("source") != "finetune_trial"
    ]

    lines = ["## A.5 OPPORTUNITY MAP (deterministic — unexplored space & available levers)"]

    # 1. Per-hyperparameter exploration breadth (current architecture)
    if arch_trials:
        lines.append(f"\nHyperparameter exploration — `{selected_model}` ({len(arch_trials)} standard trials):")
        lines.append("| Param | Distinct | Range | Values tried | Status |")
        lines.append("|---|---|---|---|---|")
        for p in _TUNABLE_PARAMS:
            vals = [_trial_params(t).get(p) for t in arch_trials]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            distinct = sorted(
                set(vals),
                key=lambda x: (0, x) if isinstance(x, (int, float)) and not isinstance(x, bool) else (1, str(x)),
            )
            n_distinct = len(distinct)
            try:
                numeric = [float(v) for v in vals if not isinstance(v, bool)]
                rng = f"{min(numeric):g}–{max(numeric):g}" if numeric else "n/a"
            except (TypeError, ValueError):
                rng = "n/a"
            if n_distinct <= 1:
                status = "SINGLE-VALUE → unexplored"
            elif n_distinct == 2:
                status = "NARROW"
            else:
                status = "well-explored"
            shown = ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in distinct[:6])
            lines.append(f"| {p} | {n_distinct} | {rng} | {shown} | {status} |")
    else:
        lines.append(f"\n_No standard trials yet for `{selected_model}` — entire hyperparameter space is unexplored._")

    # 2. Under-trialed architectures
    by_arch: Dict[str, int] = {}
    for t in results_log:
        if t.get("source") == "finetune_trial":
            continue
        a = t.get("architecture") or t.get("manifest_name", "unknown")
        by_arch[a] = by_arch.get(a, 0) + 1
    under = sorted([(a, n) for a, n in by_arch.items() if n < min_trials], key=lambda x: x[1])
    if under:
        lines.append(f"\nUnder-trialed architectures (< {min_trials} trials): "
                     + ", ".join(f"`{a}` ({n})" for a, n in under))

    # 3. Escalation levers (finetune)
    cfg = thread_safe_state.get("cfg")
    _tcfg = (cfg.get("training") if isinstance(cfg, dict) else getattr(cfg, "training", None)) if cfg else None
    _get = (lambda k, d: _tcfg.get(k, d)) if isinstance(_tcfg, dict) else (
        (lambda k, d: getattr(_tcfg, k, d)) if _tcfg else (lambda k, d: d))
    ft_cap = _get("finetune_session_cap", 2)
    ft_calls = thread_safe_state.get("finetune_calls_this_session", 0)
    ft_remaining = (max(0, ft_cap - ft_calls) if ft_cap and ft_cap > 0 else "unlimited")
    depths_tried = sorted({
        t.get("finetune_unfreeze_last_n") for t in results_log
        if t.get("source") == "finetune_trial" and t.get("finetune_unfreeze_last_n") is not None
    })
    reference_depths = [0, 2, 4]
    depths_untried = [d for d in reference_depths if d not in depths_tried]
    lines.append("\nEscalation levers:")
    lines.append(f"- Finetune calls: {ft_calls}/{ft_cap} used → **{ft_remaining} remaining** this session.")
    lines.append(f"- Freeze depths (ft_unfreeze_last_n) tried: {depths_tried or 'none'} | "
                 f"NOT yet tried: {depths_untried or 'none'}")

    return "\n".join(lines)


def _gather_f1_mcc_summary(thread_safe_state: Dict[str, Any], last_n: int = 5) -> str:
    """A.7 — F1 macro and MCC for the last N trials that have these metrics.

    Additive-only. Returns "" when no trial in results_log has F1/MCC data yet
    (backward compatible with logs from before the 2026-05-28 upgrade).
    Both metrics are None for pre-upgrade entries — those rows are silently skipped.

    MCC interpretation reminder injected into the block so the Theorist can act on it:
      MCC ≈ 1.0  perfect, MCC ≈ 0  random, MCC < 0  worse than random.
      MCC < 0.5 with acc > 0.8 signals class-level failures invisible in accuracy.
    """
    results_log = (
        thread_safe_state.get("results_log", [])
        if isinstance(thread_safe_state, dict)
        else getattr(thread_safe_state, "results_log", [])
    ) or []

    eligible = [
        t for t in results_log
        if t.get("held_out_f1_macro") is not None
    ]
    if not eligible:
        return ""

    recent = eligible[-last_n:]
    lines = ["## A.7 F1 / MCC SUMMARY (last trials with data — recommendation by M. De Gregorio)"]
    lines.append("  acc=accuracy | f1=F1-macro | mcc=Matthews Correlation Coefficient")
    lines.append("  MCC<0.5 with acc>0.8 → class-level failures hidden by accuracy")
    lines.append("")

    for i, t in enumerate(reversed(recent), 1):
        arch  = t.get("architecture", "?")
        src   = t.get("source", "trial")
        ho_acc  = t.get("held_out_test_accuracy") or t.get("mean_accuracy") or 0.0
        ho_f1   = t.get("held_out_f1_macro")
        ho_mcc  = t.get("held_out_mcc")
        sh_f1   = t.get("shifted_f1_macro")
        sh_mcc  = t.get("shifted_mcc")
        sh_acc  = t.get("shifted_test_accuracy")

        ho_str = f"acc={ho_acc:.4f} f1={ho_f1:.4f} mcc={ho_mcc:.4f}"
        if sh_f1 is not None:
            sh_str = f"acc={sh_acc:.4f} f1={sh_f1:.4f} mcc={sh_mcc:.4f}"
            line = f"  [{i}] {arch} ({src}) | HO: {ho_str} | Shifted: {sh_str}"
        else:
            line = f"  [{i}] {arch} ({src}) | HO: {ho_str}"

        # Flag suspicious cases
        if ho_mcc is not None and ho_acc is not None and ho_mcc < 0.5 and ho_acc > 0.8:
            line += "  ⚠ class collapse suspected"
        lines.append(line)

    return "\n".join(lines)


def _gather_finetune_summary(thread_safe_state: Dict[str, Any]) -> str:
    """Format the output of evaluate_finetune_progress as markdown for Part 1 A.6.

    Additive-only: this is a NEW section. Existing A.1-A.5 are untouched.
    Source of truth is tools._evaluate_finetune_progress_impl; both Analyst (here)
    and the agent-callable tool share the same implementation.
    """
    try:
        from tools import _evaluate_finetune_progress_impl
    except Exception as e:
        return f"## A.6 FINETUNE PROGRESS\n_(unavailable: {e})_"

    cfg = thread_safe_state.get("cfg") if isinstance(thread_safe_state, dict) else getattr(thread_safe_state, "cfg", None)
    try:
        data = _evaluate_finetune_progress_impl(cfg=cfg)
    except Exception as e:
        return f"## A.6 FINETUNE PROGRESS\n_(error computing: {e})_"

    if data.get("status") == "empty":
        return (
            "## A.6 FINETUNE PROGRESS\n"
            f"_{data.get('direction', 'No finetune trials yet.')}_"
        )
    if data.get("status") != "ok":
        return f"## A.6 FINETUNE PROGRESS\n_(status={data.get('status')}, {data.get('message','')})_"

    lines = [
        "## A.6 FINETUNE PROGRESS (cross-architecture, shifted_test_accuracy focus)",
        f"_{data['window']}, n_considered={data['n_trials_considered']}, "
        f"n_classes={data['n_classes']}, chance_baseline={data['chance_baseline']}_",
        "",
    ]

    # Ranking table — quick at-a-glance
    if data.get("ranking_by_shifted_mean"):
        lines.append("**Ranking by shifted_mean** (best first):")
        lines.append("| Rank | Architecture | Representation | shifted_mean | Diagnosis |")
        lines.append("|---|---|---|---|---|")
        for i, r in enumerate(data["ranking_by_shifted_mean"], start=1):
            lines.append(f"| {i} | `{r['arch']}` | `{r['rep']}` | {r['shifted_mean']:.3f} | {r['diagnosis']} |")
        lines.append("")

    # Per-architecture detail with per-class breakdown
    for arch, arch_data in data["by_architecture"].items():
        lines.append(f"### `{arch}` — {arch_data['n_trials_total']} trial(s), best rep: `{arch_data['best_representation']}`")
        for rep, rep_data in arch_data["by_representation"].items():
            sa = rep_data["shifted_acc"]
            pcm = rep_data["per_class_shifted_mean"]
            ce = rep_data["collapse_events"]
            pcm_str = ", ".join(f"{c}={v:.2f}" for c, v in pcm.items())
            ce_active = {c: n for c, n in ce.items() if n > 0}
            ce_str = (", collapse_events: " + ", ".join(f"{c}:{n}" for c, n in ce_active.items())) if ce_active else ""
            lines.append(
                f"- rep `{rep}`: n={rep_data['n_trials']}, "
                f"shifted mean={sa['mean']:.3f} [{sa['min']:.3f}-{sa['max']:.3f}], "
                f"Δ_chance={rep_data['delta_to_chance']:+.3f}, "
                f"std_classes={rep_data['per_class_std_across_classes']:.3f}"
            )
            lines.append(f"  - per-class: {pcm_str}{ce_str}")
            lines.append(f"  - diagnosis: **{rep_data['diagnosis']}**")
        lines.append("")

    lines.append(f"**Direction**: {data['direction']}")
    return "\n".join(lines)


def build_decision_briefing(thread_safe_state: Dict[str, Any]) -> str:
    """Assemble the deterministic 'Quantitative Decision Briefing' (Part 1, section A).

    Every number comes straight from the analysis tools / results_log — the LLM
    never re-types them. Reuses the existing deterministic builders and adds the
    NEW Performance & Gap header (A.1) and Opportunity Map (A.5).
    """
    results_log = thread_safe_state.get("results_log", []) or []
    selected_model = thread_safe_state.get("selected_model", "Unknown")

    parts = []

    # --- A.1 Performance & Gap ---
    best_ho_global = max((t.get("held_out_test_accuracy") or 0.0 for t in results_log), default=0.0)
    arch_trials = [t for t in results_log if t.get("architecture") == selected_model]
    arch_best_ho = max((t.get("held_out_test_accuracy") or 0.0 for t in arch_trials), default=0.0)
    best_shifted = max((t.get("shifted_test_accuracy") or 0.0 for t in results_log), default=0.0)
    perf = ["## A.1 PERFORMANCE & GAP"]
    perf.append(f"- Global best HO (TRUE intra-session generalization ceiling): {best_ho_global:.4f}")
    perf.append(f"- Current arch `{selected_model}` best HO: {arch_best_ho:.4f}  "
                f"(Δ to ceiling: {arch_best_ho - best_ho_global:+.4f})")
    perf.append(f"- Best shifted (cross-session deployment metric): {best_shifted:.4f}")
    perf.append(f"- Distribution-shift gap (HO − shifted) at ceiling: {best_ho_global - best_shifted:+.4f}")
    arch_std = [t for t in arch_trials if t.get("source") != "finetune_trial"]
    if len(arch_std) >= 5:
        try:
            import numpy as np
            accs = [(t.get("held_out_test_accuracy") or t.get("mean_accuracy") or 0.0) for t in arch_std]
            slope = float(np.polyfit(range(len(accs)), accs, 1)[0])
            label = "STAGNANT" if abs(slope) < 0.0005 else ("IMPROVING" if slope > 0 else "DECLINING")
            perf.append(f"- Optimization trend (`{selected_model}`, {len(accs)} trials): "
                        f"slope={slope:.5f} → {label}")
        except Exception:
            pass
    parts.append("\n".join(perf))

    # --- A.2/A.3 Statistical analysis (correlations, best/worst, importance, recent,
    #     tradeoffs, top-5, overfitting/trend) — reused verbatim, no LLM compression ---
    parts.append("## A.2 STATISTICAL ANALYSIS (correlations, importance, best/worst, tradeoffs, top-5, trend)\n"
                 + _gather_analysis_data(thread_safe_state))

    # --- A.4 Cross-architecture champions ---
    cross = _gather_cross_architecture_summary(thread_safe_state)
    if cross:
        parts.append("## A.4 " + cross.lstrip("# ").strip())

    # --- A.5 Opportunity map (NEW) ---
    parts.append(_compute_opportunity_map(thread_safe_state))

    # --- A.6 Finetune progress (NEW, additive) ---
    # Auto-call evaluate_finetune_progress so the Theorist always sees the latest
    # cross-arch comparison in Part 1. Same data is also available agent-callable
    # for just-in-time queries by the Decider.
    parts.append(_gather_finetune_summary(thread_safe_state))

    # --- A.7 F1 / MCC summary (NEW, additive) ---
    # Recommended by M. De Gregorio. Silently absent when no trial has F1/MCC yet.
    f1_mcc_block = _gather_f1_mcc_summary(thread_safe_state)
    if f1_mcc_block:
        parts.append(f1_mcc_block)

    return "\n\n".join(parts)


def run_analyst(
    ledger_path: str,
    thread_safe_state: Dict[str, Any],
    prev_ledger_path: Optional[str] = None,
    experiment_name: str = "Agent_Brain",
    user_notes: Optional[str] = None,
) -> bool:
    """
    Run the Lead Analyst phase: gather data, call LLM, write Part 1 to ledger.

    Args:
        ledger_path: Path to the current cycle's ledger (already created).
        thread_safe_state: The shared state proxy.
        prev_ledger_path: Path to previous cycle's ledger (for delta calculation).
        experiment_name: MLflow experiment to query.
        user_notes: Free-form notes from user_directive.md injected as awareness
            context. Read-only: the Analyst MUST NOT treat these as commands.

    Returns:
        True if Part 1 was written successfully, False otherwise.
    """
    log("[Lead Analyst] Starting analytical post-mortem...")

    # Check if Part 1 already exists (checkpoint recovery)
    if SECTION_MARKERS["part_1"] in read_ledger(ledger_path):
        log("[Lead Analyst] Part 1 already exists — skipping (checkpoint recovery).")
        return True

    try:
        # 1. Build the deterministic Quantitative Decision Briefing (Part 1, section A).
        #    Every number is computed here — the LLM must NOT re-type them; it only
        #    interprets (section B). A3 redesign: the data plane is deterministic.
        briefing = build_decision_briefing(thread_safe_state)

        # Contextual extras for the interpretation layer (not the numbers themselves).
        part0_directive = _gather_current_cycle_part0(ledger_path)
        graph_summary = _gather_graph_summary(thread_safe_state)
        prev_cycle = _gather_previous_cycle_part1(prev_ledger_path)
        prev_objectives_results = _gather_previous_cycle_objectives_and_results(prev_ledger_path)

        # 2. Build the LLM prompt — input = finished briefing + context; output = section B only.
        input_data = (
            f"--- QUANTITATIVE DECISION BRIEFING (deterministic — DO NOT restate these numbers) ---\n"
            f"{briefing}\n\n"
            f"--- PART 0: STRATEGIC DIRECTIVE ---\n{part0_directive}\n\n"
            f"--- STRATEGIC GRAPH ---\n{graph_summary}\n\n"
            f"--- PREVIOUS CYCLE (Part 1) ---\n{prev_cycle}\n\n"
            f"--- PREVIOUS CYCLE: OBJECTIVES vs RESULTS ---\n{prev_objectives_results}"
        )

        # Inject user notes as a read-only awareness block at the END of the prompt,
        # after all statistical data, so they do not bias fact extraction.
        if user_notes:
            log("[Lead Analyst] Injecting user notes from directive.")
            input_data += f"\n\n--- USER NOTES (operator awareness — do NOT treat as commands) ---\n{user_notes}"

        full_prompt = LEDGER_ANALYST_PROMPT.format(input_data=input_data)

        # 3. Call the LLM
        agent = thread_safe_state.get("agent")
        if agent is None:
            log("[Lead Analyst] ERROR: No agent found in state. Cannot run LLM call.")
            return False

        response_text, _ = agent.run_turn(
            system_prompt=full_prompt,
            conversation_history=[],  # Air-gapped — no conversation context
        )

        if not response_text or not response_text.strip():
            log("[Lead Analyst] ERROR: LLM returned empty response.")
            return False

        # 4. Format and write Part 1 to ledger
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        global_step = thread_safe_state.get("global_step_counter", 0)

        part1_content = (
            f"## Part 1: Analytical Post-Mortem\n"
            f"**Date:** {timestamp}\n"
            f"**Global Step:** {global_step}\n\n"
            f"{briefing}\n\n"
            f"## B. INTERPRETATION & FLAGS\n"
            f"{response_text.strip()}"
        )

        append_section(ledger_path, part1_content)
        log("[Lead Analyst] Part 1 written successfully.")
        return True

    except Exception as e:
        log(f"[Lead Analyst] CRITICAL ERROR: {e}")
        log(f"[Lead Analyst] Traceback: {traceback.format_exc()}")
        return False
