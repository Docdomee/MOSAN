# ledger_schema.py
"""
Single Ledger Architecture — Ledger File Schema & Utilities.

The ledger is the sole communication channel between the three agents
(Lead Analyst, Theorist, Decider). This module handles creation, reading,
validation, and atomic appending of ledger sections.

Contract: Ledgers are append-only. No agent overwrites another's section.
"""

import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Optional

from state_manager import PERSISTENT_PATHS
from ui_logger import log

# Default workspace location — inside project root
WORKSPACE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "workspace"
)

# Section markers used for checkpoint detection (order matters)
SECTION_MARKERS = {
    "part_0": "## Part 0: Strategic Continuity (Context Injection)",
    "part_1": "## Part 1: Analytical Post-Mortem",
    "part_2": "## Part 2: Strategic Context",
    "hypothesis": "**Hypothesis:**",
    "objectives": "**Semantic Objectives:**",
    "budget": "**Step Budget:**",
    "part_3": "## Part 3: Strategic Plan & Objectives",
    "part_4": "## Part 4: Decider's Execution Log",
}


def _ensure_workspace(workspace_dir: str = WORKSPACE_DIR) -> str:
    """Ensure workspace directory exists. Returns the path."""
    os.makedirs(workspace_dir, exist_ok=True)
    return workspace_dir


def create_ledger(cycle_number: int, workspace_dir: str = WORKSPACE_DIR, preamble: Optional[str] = None) -> str:
    """
    Create a new cycle_N_ledger.md with the cycle header.

    Args:
        cycle_number: The cycle number for this ledger.
        workspace_dir: Directory where ledger files live.
        preamble: Optional text from the previous cycle's strategic summarizer.

    Returns:
        Absolute path to the newly created ledger file.
    """
    _ensure_workspace(workspace_dir)
    filename = f"cycle_{cycle_number}_ledger.md"
    filepath = os.path.join(workspace_dir, filename)

    header = (
        f"# Cycle {cycle_number} Ledger\n"
        f"**Created:** {datetime.now(timezone.utc).isoformat()}\n\n"
    )

    if preamble:
        header += f"## Part 0: Strategic Continuity (Context Injection)\n\n{preamble.strip()}\n\n---\n\n"

    # Atomic write: write to temp file, then rename
    _atomic_write(filepath, header)
    log(f"[Ledger] Created: {filename}")
    return filepath


def append_section(ledger_path: str, content: str) -> None:
    """
    Atomically append a section to the ledger.

    Reads existing content, appends new content with a separator,
    then performs an atomic write (temp file + rename).

    Args:
        ledger_path: Path to the ledger file.
        content: The section content to append (should include its own ## header).
    """
    existing = read_ledger(ledger_path)
    updated = existing.rstrip("\n") + "\n\n---\n\n" + content.strip() + "\n"
    _atomic_write(ledger_path, updated)
    log(f"[Ledger] Appended section to {os.path.basename(ledger_path)}")


def append_raw(ledger_path: str, text: str) -> None:
    """
    Atomically append raw text to the ledger WITHOUT a section separator.

    Unlike append_section (which inserts a '---' divider and starts a new block),
    this appends text *inside* the current section. Used for incremental,
    append-only writes such as the Decider's per-step Part 4 log, so an interrupt
    mid-cycle still leaves every completed step on disk.
    """
    existing = read_ledger(ledger_path)
    updated = existing.rstrip("\n") + "\n" + text.rstrip("\n") + "\n"
    _atomic_write(ledger_path, updated)


def read_ledger(ledger_path: str) -> str:
    """Read full ledger content. Returns empty string if file doesn't exist."""
    if not os.path.exists(ledger_path):
        return ""
    with open(ledger_path, "r", encoding="utf-8") as f:
        return f.read()


def get_completed_sections(ledger_path: str) -> list[str]:
    """
    Parse which section markers exist in the ledger.

    Returns:
        List of marker keys (e.g., ["part_1", "part_2", "hypothesis"]) that
        are present in the ledger content.
    """
    content = read_ledger(ledger_path)
    found = []
    for key, marker in SECTION_MARKERS.items():
        if marker in content:
            found.append(key)
    return found


def get_latest_ledger(workspace_dir: str = WORKSPACE_DIR) -> tuple[int, Optional[str]]:
    """
    Find the highest cycle number ledger in the workspace.

    Returns:
        (cycle_number, filepath) — or (0, None) if no ledgers exist.
    """
    _ensure_workspace(workspace_dir)
    pattern = re.compile(r"^cycle_(\d+)_ledger\.md$")
    max_cycle = 0
    max_path = None

    for fname in os.listdir(workspace_dir):
        match = pattern.match(fname)
        if match:
            cycle_num = int(match.group(1))
            if cycle_num > max_cycle:
                max_cycle = cycle_num
                max_path = os.path.join(workspace_dir, fname)

    return max_cycle, max_path


def parse_section(ledger_path: str, section_name: str) -> Optional[str]:
    """
    Extract content of a specific section by its ## header name.

    Args:
        ledger_path: Path to the ledger.
        section_name: The header text after '## ' (e.g., "Part 1: Analytical Post-Mortem").

    Returns:
        The section content (excluding the header line itself), or None if not found.
    """
    content = read_ledger(ledger_path)
    # Find section start
    header_pattern = re.compile(
        rf"^## {re.escape(section_name)}\s*$", re.MULTILINE
    )
    match = header_pattern.search(content)
    if not match:
        return None

    start = match.end()

    # Find next section (## header or --- separator before next ##) or end of file
    next_section = re.compile(r"^---\s*$\s*^## ", re.MULTILINE)
    next_match = next_section.search(content, start)
    if next_match:
        end = next_match.start()
    else:
        end = len(content)

    return content[start:end].strip()


def _atomic_write(filepath: str, content: str) -> None:
    """
    Write content to filepath atomically.
    Writes to a temp file in the same directory, then renames.
    On Windows, os.rename can fail if target exists, so we use os.replace.
    """
    dir_path = os.path.dirname(filepath)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, filepath)
    except Exception as e:
        # Cleanup temp file on failure
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Atomic write failed for {filepath}: {e}") from e
