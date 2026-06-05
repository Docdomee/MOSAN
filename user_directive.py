# user_directive.py
"""
Single Ledger Architecture — Phase 3: User Directive (Async Mailbox).

The user_directive.md file is the human-in-the-loop control mechanism.
Users edit it at any time; the epoch runner reads it at every cycle boundary.

Contract: The system reads but never writes to user_directive.md.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from ui_logger import log


DIRECTIVE_FILENAME = "user_directive.md"

# Whitelist of valid status values. Anything else is a typo and falls back to
# "running" with an explicit warning so the user knows the file is malformed.
_VALID_STATUSES = {"running", "paused", "stopped"}


@dataclass
class UserDirective:
    """Parsed user directive — controls epoch runner and agent behavior."""
    max_epochs: int = 50
    pause_after_epoch: Optional[int] = None
    status: str = "running"            # "running" | "paused" | "stopped"
    priority_override: Optional[str] = None   # Injected into Theorist prompts
    constraints: List[str] = field(default_factory=list)  # Injected into Budgeting + Decider
    notes: Optional[str] = None        # General awareness for Analyst

    @classmethod
    def default(cls) -> "UserDirective":
        """Return a permissive default directive (no constraints)."""
        return cls()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_int_field(content: str, field_name: str, default: Optional[int] = None) -> Optional[int]:
    """
    Extract an integer field like '**Max Epochs:** 10' from markdown.

    Also handles the broken two-line format that can appear when users edit
    the file without a trailing space before the value:
        **Pause After Epoch:19
        Status:** running
    In this case the value is run together with the key on the same line.
    """
    # Primary: correct format — '**Field:** value'
    primary = re.compile(rf"\*\*{re.escape(field_name)}:\*\*\s*(\d+)", re.IGNORECASE)
    match = primary.search(content)
    if match:
        return int(match.group(1))

    # Fallback: broken format — '**Field:VALUE' (colon immediately followed by digits,
    # no closing '**' before the value).  This is purely defensive; the canonical
    # user_directive.md should always use the correct format.
    fallback = re.compile(rf"\*\*{re.escape(field_name)}:(\d+)", re.IGNORECASE)
    match = fallback.search(content)
    if match:
        log(f"[UserDirective] WARNING: '{field_name}' parsed via fallback pattern — "
            f"please fix the format in {DIRECTIVE_FILENAME} (**{field_name}:** value).")
        return int(match.group(1))

    return default


def _parse_field(content: str, field_name: str, default: str = "") -> str:
    """Extract a text field like '**Status:** running' from markdown."""
    pattern = re.compile(rf"\*\*{re.escape(field_name)}:\*\*\s*(.+)", re.IGNORECASE)
    match = pattern.search(content)
    if match:
        return match.group(1).strip()
    return default


def _parse_section_content(content: str, section_name: str) -> Optional[str]:
    """Extract free-text content under a ## heading, up to the next ## or EOF."""
    pattern = re.compile(
        rf"^## {re.escape(section_name)}\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    match = pattern.search(content)
    if not match:
        return None

    start = match.end()
    # Find next ## header or EOF
    next_header = re.compile(r"^## ", re.MULTILINE)
    next_match = next_header.search(content, start)
    end = next_match.start() if next_match else len(content)

    text = content[start:end].strip()
    return text if text else None


def _parse_bullet_list(content: str, section_name: str) -> List[str]:
    """Extract a bullet list from a ## section."""
    section_text = _parse_section_content(content, section_name)
    if not section_text:
        return []
    bullets = re.findall(r"^[-*]\s+(.+)", section_text, re.MULTILINE)
    return [b.strip() for b in bullets if b.strip()]


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------

def read_user_directive(workspace_dir: str) -> UserDirective:
    """
    Parse workspace/user_directive.md into a UserDirective object.

    If the file doesn't exist, returns a permissive default (system runs freely).
    """
    path = os.path.join(workspace_dir, DIRECTIVE_FILENAME)

    if not os.path.exists(path):
        log("[UserDirective] No user_directive.md found — using defaults.")
        return UserDirective.default()

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        log(f"[UserDirective] Error reading {path}: {e} — using defaults.")
        return UserDirective.default()

    raw_status = _parse_field(content, "Status", default="running").lower()
    if raw_status not in _VALID_STATUSES:
        log(
            f"[UserDirective] WARNING: Unrecognised status '{raw_status}' in {DIRECTIVE_FILENAME}. "
            f"Valid values: {sorted(_VALID_STATUSES)}. Defaulting to 'running'."
        )
        raw_status = "running"

    directive = UserDirective(
        max_epochs=_parse_int_field(content, "Max Epochs", default=50),
        pause_after_epoch=_parse_int_field(content, "Pause After Epoch", default=None),
        status=raw_status,
        priority_override=_parse_section_content(content, "Priority Override"),
        constraints=_parse_bullet_list(content, "Constraints"),
        notes=_parse_section_content(content, "Notes"),
    )

    log(
        f"[UserDirective] Loaded: status={directive.status}, "
        f"max_epochs={directive.max_epochs}, "
        f"pause_after_epoch={directive.pause_after_epoch}, "
        f"constraints={len(directive.constraints)}, "
        f"has_priority={directive.priority_override is not None}, "
        f"has_notes={directive.notes is not None}"
    )

    return directive
