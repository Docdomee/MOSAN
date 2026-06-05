"""
Architecture Registry — scans processed_data/custom_architectures/ on every
Theorist invocation, classifies each architecture by parsing its .md changelog,
filters by current dimensionality, and ranks by score.

Public entry point: `build_innovation_context(thread_safe_state, selected_model)`.
Output is a markdown string injected into the Theorist prompt's `innovation_context`
field (replacing the old single-cycle behaviour at theorist.py:599-610).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from state_manager import PERSISTENT_PATHS

# ---------------------------------------------------------------------------
# Tunable scoring weights (Slot B — Reusable Trained Architectures).
# Edit here to change behaviour; do NOT hardcode values inside _score_entry.
# ---------------------------------------------------------------------------
WEIGHTS = {
    "success_quasi_champion":      100,   # HO >= champion - 0.05
    "system_error_pending_retry":   80,   # importlib / OOM / manifest_not_found
    "passed_validation_no_train":   60,   # only if Pending leaks into Slot B
    "success_decent":               50,   # HO >= champion - 0.10
    "ambiguous_failure":            40,   # FAILED TRAINING but unclear cause
    "fresh_24h":                    50,
    "fresh_7d":                     25,
    "stale_30d":                   -20,
    "exact_rep_match":              30,
    "retry_count_3plus":           -40,
}

SLOT_B_CAP = 5
EXCLUDE_HO_DELTA = 0.10  # entries with HO < champion - 0.10 are excluded from Slot B

LIBRARY_DIR_REL = os.path.join("cluster", "library", "architectures")

# ---------------------------------------------------------------------------
# Failure mode regexes — order matters: system check first, then architectural.
# ---------------------------------------------------------------------------
_RE_SYS_FAILURE = re.compile(
    r"\[SYSTEM_ERROR\]"
    r"|\bimportlib\b"
    r"|ModuleNotFoundError"
    r"|OutOfMemoryError|OOM\b|ResourceExhaustedError"
    r"|manifest.{0,40}not\s*found"
    r"|FileNotFoundError.{0,80}manifest"
    r"|Killed process|SIGKILL",
    re.IGNORECASE,
)

_RE_ARCH_FAILURE = re.compile(
    r"\bNaN\b"
    r"|shape mismatch"
    r"|incompatible shapes"
    r"|Negative dimension"
    r"|InvalidArgumentError.{0,80}shape"
    r"|Dimensions must be equal",
    re.IGNORECASE,
)

# Section header in changelog .md files. Stable schema written by
# tools.append_to_architecture_changelog (see tools.py:3519).
_RE_SECTION = re.compile(
    r"^##\s+(PASSED VALIDATION|FAILED VALIDATION|SUCCESS|FAILED TRAINING|DEBUG FIX APPLIED)"
    r"\s*\(Attempt\s+(\d+)\)",
    re.MULTILINE,
)

# Conv layer detection — first hit wins; matches Conv1D/2D/3D and variants
# like Conv2DTranspose, SeparableConv1D, etc.
_RE_CONV = re.compile(r"\bConv([123])D\b")

# Filename hint fallback. Matches:
#   - library convention suffix: standard_cnn_3d.py
#   - explicit dim tokens anywhere: 1D_CNN_baseline.py, gaf_2d_resnet.py
_RE_NAME_DIM = re.compile(r"(?:^|[_\-])([123])D(?:[_\-]|\.py$)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ChangelogParse:
    """Parsed view of a single architecture's .md changelog."""
    sections: list[tuple[str, int, str]] = field(default_factory=list)  # [(kind, attempt_n, body)]
    last_kind: Optional[str] = None
    last_body: Optional[str] = None
    failed_training_count: int = 0


@dataclass
class ArchEntry:
    name: str
    py_path: Path
    md_path: Optional[Path]
    dimensionality: Literal["1d", "2d", "3d", "unknown"]
    status: Literal["pending", "success", "sys_fail", "arch_fail", "broken", "none"]
    error_text: Optional[str]
    ho_acc: Optional[float]
    manifest_name: Optional[str]
    representation_type: Optional[str]
    mtime: float
    retry_count: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_innovation_context(thread_safe_state: dict, selected_model: str) -> str:
    """
    Build the Theorist's `innovation_context` block from the on-disk registry.

    Replaces the original simple builder at theorist.py:599-610. Returns a
    legacy-compatible fallback string when the registry yields nothing, so
    callers don't need branching logic.
    """
    custom_dir = Path(PERSISTENT_PATHS["custom_architectures_dir"])
    project_root = Path(PERSISTENT_PATHS["processed_data_dir"]).parent
    library_dir = project_root / LIBRARY_DIR_REL

    repr_tag = _derive_repr_tag(selected_model)

    entries = _scan_architectures(custom_dir) if custom_dir.exists() else []

    results_log = thread_safe_state.get("results_log", []) or []
    champion_ho = _compute_champion_ho(results_log, selected_model)

    last_innovation = thread_safe_state.get("last_innovation_result") or {}
    last_innovation_file = (
        last_innovation.get("architecture_file")
        if isinstance(last_innovation, dict) else None
    )

    # Slot A: pending entries (validated but not yet trained or fail-trained).
    # Allow dimensionality matching repr_tag OR "unknown" (transformers etc.).
    slot_a = [
        e for e in entries
        if e.status == "pending"
        and e.dimensionality in (repr_tag, "unknown")
        and e.name != last_innovation_file  # dedup with last-cycle block
    ]
    slot_a.sort(key=lambda e: e.mtime, reverse=True)

    # Slot B: reusable trained entries — strict dimensionality match.
    slot_b_candidates = [
        e for e in entries
        if e.status in ("success", "sys_fail", "arch_fail")
        and e.dimensionality == repr_tag
    ]
    scored: list[tuple[int, ArchEntry]] = []
    for e in slot_b_candidates:
        s = _score_entry(e, champion_ho, selected_model, last_innovation_file)
        if s is not None:
            scored.append((s, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    slot_b = scored[:SLOT_B_CAP]

    # Slot C: library templates filtered by repr_tag.
    slot_c = _list_library_templates(library_dir, repr_tag)

    # Last-cycle block — preserve legacy behaviour when status == "success".
    last_cycle_block = ""
    if isinstance(last_innovation, dict) and last_innovation.get("status") == "success":
        arch_file = last_innovation.get("architecture_file", "unknown")
        rep_type = last_innovation.get("representation_type", "unknown")
        last_cycle_block = (
            f"The Innovation Team successfully built a new architecture last cycle.\n"
            f"- File: `{arch_file}`\n"
            f"- Representation type: `{rep_type}`\n"
            f"**Your first objective MUST be to test and optimize this new architecture.**"
        )

    return _format_for_prompt(
        slot_a=slot_a,
        slot_b=slot_b,
        slot_c=slot_c,
        last_cycle_block=last_cycle_block,
        repr_tag=repr_tag,
        selected_model=selected_model,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _derive_repr_tag(selected_model: str) -> str:
    """Mirror of agent.py:1322-1327 — convert representation_type to dim tag."""
    s = (selected_model or "").upper()
    if "2D" in s or "IMAGE" in s:
        return "2d"
    if "3D" in s or "VIDEO" in s:
        return "3d"
    return "1d"


def _scan_architectures(custom_dir: Path) -> list[ArchEntry]:
    """Walk *.py in custom_dir, build one ArchEntry per file with sibling .md."""
    entries: list[ArchEntry] = []
    for py_path in sorted(custom_dir.glob("*.py")):
        md_path = py_path.with_suffix(".md")
        if not md_path.exists():
            # Orphan rule: no changelog, exclude from registry entirely.
            continue

        try:
            parse = _parse_changelog_md(md_path)
        except Exception:
            continue  # malformed .md, skip silently

        try:
            dim = _detect_arch_dimensionality(py_path)
        except Exception:
            dim = "unknown"

        status = _classify_status(parse)
        ho_acc, manifest_name, rep_type = _extract_success_meta(parse)

        try:
            mtime = md_path.stat().st_mtime
        except OSError:
            mtime = py_path.stat().st_mtime if py_path.exists() else 0.0

        entries.append(ArchEntry(
            name=py_path.name,
            py_path=py_path,
            md_path=md_path,
            dimensionality=dim,
            status=status,
            error_text=parse.last_body if status in ("sys_fail", "arch_fail", "broken") else None,
            ho_acc=ho_acc,
            manifest_name=manifest_name,
            representation_type=rep_type,
            mtime=mtime,
            retry_count=parse.failed_training_count,
        ))
    return entries


def _parse_changelog_md(md_path: Path) -> ChangelogParse:
    """
    Parse a changelog .md into ordered (kind, attempt_n, body) sections.
    The last entry by file order is the most recent state — Attempt N can
    repeat across kinds (e.g. PASSED VALIDATION Attempt 1 then SUCCESS Attempt 1).
    """
    text = md_path.read_text(encoding="utf-8", errors="replace")
    matches = list(_RE_SECTION.finditer(text))
    sections: list[tuple[str, int, str]] = []
    for i, m in enumerate(matches):
        kind = m.group(1).strip()
        attempt_n = int(m.group(2))
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((kind, attempt_n, body))

    parse = ChangelogParse(sections=sections)
    if sections:
        parse.last_kind = sections[-1][0]
        parse.last_body = sections[-1][2]
    parse.failed_training_count = sum(1 for k, _, _ in sections if k == "FAILED TRAINING")
    return parse


def _detect_arch_dimensionality(py_path: Path) -> Literal["1d", "2d", "3d", "unknown"]:
    """First Conv*D match wins; fallback to filename suffix; else 'unknown'."""
    try:
        text = py_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "unknown"

    m = _RE_CONV.search(text)
    if m:
        return f"{m.group(1)}d"  # type: ignore[return-value]

    m = _RE_NAME_DIM.search(py_path.name)
    if m:
        return f"{m.group(1)}d"  # type: ignore[return-value]

    return "unknown"


def _classify_status(parse: ChangelogParse) -> Literal["pending", "success", "sys_fail", "arch_fail", "broken", "none"]:
    """
    Determine current status from the last meaningful section.
    DEBUG FIX APPLIED is transitional → look one section back.
    """
    if not parse.sections:
        return "none"

    # Walk backwards skipping transitional DEBUG FIX APPLIED markers.
    last_meaningful: Optional[tuple[str, int, str]] = None
    for kind, attempt_n, body in reversed(parse.sections):
        if kind == "DEBUG FIX APPLIED":
            continue
        last_meaningful = (kind, attempt_n, body)
        break

    if last_meaningful is None:
        # Only DEBUG FIX APPLIED entries — treat as pending.
        return "pending"

    kind, _, body = last_meaningful
    if kind == "PASSED VALIDATION":
        return "pending"
    if kind == "SUCCESS":
        return "success"
    if kind == "FAILED VALIDATION":
        # Code never even passed syntax check — broken, do NOT propose.
        return "broken"
    if kind == "FAILED TRAINING":
        mode = _classify_failure_mode(body or "")
        if mode == "architectural":
            return "arch_fail"
        # Both "system" and "ambiguous" route to sys_fail; the score uses the
        # failure mode to decide between full and reduced retry weight.
        return "sys_fail"
    return "none"


def _classify_failure_mode(error_text: str) -> Literal["system", "architectural", "ambiguous"]:
    if _RE_ARCH_FAILURE.search(error_text):
        return "architectural"
    if _RE_SYS_FAILURE.search(error_text):
        return "system"
    return "ambiguous"


def _extract_success_meta(parse: ChangelogParse) -> tuple[Optional[float], Optional[str], Optional[str]]:
    """Extract HO accuracy, manifest name, and representation_type from the
    most recent SUCCESS section's embedded JSON, if any."""
    last_success = None
    for kind, _, body in reversed(parse.sections):
        if kind == "SUCCESS":
            last_success = body
            break
    if last_success is None:
        return None, None, None

    # The body contains a JSON object after `**Final Result:**` or just inline.
    # Find first {...} block; tolerate surrounding markdown.
    brace_start = last_success.find("{")
    if brace_start == -1:
        return None, None, None

    # Walk to matching closing brace (handles nested objects).
    depth = 0
    end = -1
    for i in range(brace_start, len(last_success)):
        c = last_success[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None, None, None

    try:
        data = json.loads(last_success[brace_start:end])
    except (json.JSONDecodeError, ValueError):
        return None, None, None

    ho = data.get("held_out_test_accuracy")
    manifest = data.get("manifest_name")
    params = data.get("params") or {}
    rep_type = params.get("representation_type") or data.get("representation_type")
    return (
        float(ho) if isinstance(ho, (int, float)) else None,
        str(manifest) if manifest else None,
        str(rep_type) if rep_type else None,
    )


def _compute_champion_ho(results_log: list, selected_model: str) -> Optional[float]:
    """Best HO accuracy for current architecture; fallback to global best."""
    if not results_log:
        return None

    def _has_ho(t: dict) -> bool:
        return isinstance(t.get("held_out_test_accuracy"), (int, float))

    matched = [t for t in results_log if t.get("architecture") == selected_model and _has_ho(t)]
    if not matched:
        matched = [t for t in results_log if _has_ho(t)]
    if not matched:
        return None
    return max(float(t["held_out_test_accuracy"]) for t in matched)


def _score_entry(
    entry: ArchEntry,
    champion_ho: Optional[float],
    selected_model: str,
    exclude_arch_file: Optional[str],
) -> Optional[int]:
    """Return integer score, or None to exclude from Slot B."""
    if entry.name == exclude_arch_file:
        return None
    if entry.status in ("arch_fail", "broken"):
        return None
    if (
        champion_ho is not None
        and entry.ho_acc is not None
        and entry.ho_acc < champion_ho - EXCLUDE_HO_DELTA
    ):
        return None

    score = 0

    # Status base
    if entry.status == "success" and champion_ho is not None and entry.ho_acc is not None:
        if entry.ho_acc >= champion_ho - 0.05:
            score += WEIGHTS["success_quasi_champion"]
        elif entry.ho_acc >= champion_ho - 0.10:
            score += WEIGHTS["success_decent"]
    elif entry.status == "success":
        # No champion yet — give moderate credit so successes still beat failures.
        score += WEIGHTS["success_decent"]
    elif entry.status == "sys_fail":
        mode = _classify_failure_mode(entry.error_text or "")
        if mode == "ambiguous":
            score += WEIGHTS["ambiguous_failure"]
        else:
            score += WEIGHTS["system_error_pending_retry"]

    # Freshness
    age_days = (time.time() - entry.mtime) / 86400.0
    if age_days <= 1:
        score += WEIGHTS["fresh_24h"]
    elif age_days <= 7:
        score += WEIGHTS["fresh_7d"]
    elif age_days >= 30:
        score += WEIGHTS["stale_30d"]

    # Exact representation match
    if entry.representation_type and selected_model:
        if entry.representation_type.lower() == selected_model.lower():
            score += WEIGHTS["exact_rep_match"]

    # Retry penalty
    if entry.retry_count >= 3:
        score += WEIGHTS["retry_count_3plus"]

    return score


def _list_library_templates(library_dir: Path, repr_tag: str) -> list[str]:
    if not library_dir.exists():
        return []
    suffix = f"_{repr_tag}.py"
    return sorted(p.name for p in library_dir.glob("*.py") if p.name.endswith(suffix))


def _format_for_prompt(
    slot_a: list[ArchEntry],
    slot_b: list[tuple[int, ArchEntry]],
    slot_c: list[str],
    last_cycle_block: str,
    repr_tag: str,
    selected_model: str,
) -> str:
    if not slot_a and not slot_b and not slot_c and not last_cycle_block:
        # Backward-compatible fallback.
        return "(no new architecture built last cycle)"

    parts: list[str] = []

    # Slot A — Pending
    if slot_a:
        lines = ["## Pending Architectures (forged but not yet trained — train these first)"]
        for e in slot_a:
            age = _humanize_age(e.mtime)
            extra = " [unknown dim]" if e.dimensionality == "unknown" else ""
            lines.append(f"- `{e.name}` (validated {age} ago){extra}")
        parts.append("\n".join(lines))

    # Slot B — Reusable Trained
    if slot_b:
        lines = [
            f"## Reusable Custom Architectures for `{selected_model}` "
            f"({repr_tag}, top-{len(slot_b)} by score)"
        ]
        for rank, (score, e) in enumerate(slot_b, 1):
            age = _humanize_age(e.mtime)
            ho_str = f"HO={e.ho_acc:.3f}" if e.ho_acc is not None else "HO=N/A"
            reason = _entry_reason(e)
            lines.append(
                f"{rank}. `{e.name}` — status={e.status}, {ho_str}, {age} ago, score={score}\n"
                f"   reason: {reason}"
            )
        parts.append("\n".join(lines))

    # Slot C — Library Templates
    if slot_c:
        lines = [f"## Library Templates Available ({repr_tag})"]
        for name in slot_c:
            lines.append(f"- `cluster/library/architectures/{name}`")
        parts.append("\n".join(lines))

    # Last cycle block (legacy)
    if last_cycle_block:
        parts.append("## Last Cycle's Innovation\n" + last_cycle_block)

    return "\n\n".join(parts)


def _humanize_age(mtime: float) -> str:
    seconds = max(0.0, time.time() - mtime)
    minutes = seconds / 60.0
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{int(minutes)} min"
    hours = minutes / 60.0
    if hours < 24:
        return f"{int(hours)}h"
    days = hours / 24.0
    if days < 30:
        return f"{int(days)}d"
    return f"{int(days/30)}mo"


def _entry_reason(e: ArchEntry) -> str:
    if e.status == "success":
        return "previously trained successfully; candidate for refinement or ablation."
    if e.status == "sys_fail":
        return "previously failed for system error (importlib/OOM/manifest); retry recommended."
    if e.status == "arch_fail":
        return "previously failed for architectural reason; not recommended."
    if e.status == "pending":
        return "validated but never trained."
    return "unclassified."
