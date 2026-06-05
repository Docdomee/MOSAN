# saturation_logger.py
#
# Append-only JSONL log for offline analysis of per-lens saturation events.
# Used by StrategicAdvisorTeam.generate_advice() whenever a lens saturation
# is detected. Essential for EXP-A2 lens-ablation analysis.

import os
import json
from typing import Any, Dict

from state_manager import PERSISTENT_PATHS


def log_saturation_event(event_dict: Dict[str, Any]) -> None:
    """Append a saturation event as one JSONL line. Best-effort: never raises."""
    try:
        log_path = os.path.join(
            PERSISTENT_PATHS["processed_data_dir"], "logs", "saturation_events.jsonl"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_dict, default=str) + "\n")
    except Exception:
        # Telemetry must never crash the agent.
        pass
