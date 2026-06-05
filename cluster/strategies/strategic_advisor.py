
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
import numpy as np
from .metric_specialists import AccuracySpecialist, StabilitySpecialist, NoveltySpecialist, EfficiencySpecialist

if TYPE_CHECKING:
    from graph_memory import StrategicGraphMemory

# Default saturation_info returned when detection is disabled / unavailable.
_SATURATION_DISABLED = {
    "is_saturated": False,
    "lens": "DISABLED",
    "metric_name": "",
    "metric_value": 0.0,
    "threshold": 0.0,
    "n_graph_nodes": 0,
}

# Map dominant lens-weight key -> canonical lens name.
# Tie-break priority: nov > eff > stab > acc (Explorer over Engineer over Sage over Hawk).
_LENS_KEY_PRIORITY = ("nov", "eff", "stab", "acc")
_LENS_KEY_TO_NAME = {
    "acc": "HAWK",
    "stab": "SAGE",
    "nov": "EXPLORER",
    "eff": "ENGINEER",
}

class StrategicContext:
    """
    Defines the 'Lens' through which the agent views the world.
    Weights sum to 1.0 ideally, but absolute magnitude matters for prioritization.
    """
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights # e.g. {'acc': 0.8, 'stab': 0.2, ...}

    @staticmethod
    def get_hawk_lens():
        """Focus on Accuracy."""
        return StrategicContext({'acc': 0.9, 'stab': 0.1, 'nov': 0.0, 'eff': 0.0})

    @staticmethod
    def get_sage_lens():
        """Focus on Stability/Robustness."""
        return StrategicContext({'acc': 0.2, 'stab': 0.8, 'nov': 0.0, 'eff': 0.0})

    @staticmethod
    def get_explorer_lens():
        """Focus on Novelty."""
        return StrategicContext({'acc': 0.3, 'stab': 0.0, 'nov': 0.7, 'eff': 0.0})

    @staticmethod
    def get_engineer_lens():
        """Focus on Efficiency (Latency/Memory)."""
        return StrategicContext({'acc': 0.4, 'stab': 0.1, 'nov': 0.0, 'eff': 0.5})

class ParetoRanker:
    """
    Arbitrates between conflicting specialist advice.
    """
    def __init__(self):
        self.specs = {
            'acc': AccuracySpecialist(),
            'stab': StabilitySpecialist(),
            'nov': NoveltySpecialist(),
            'eff': EfficiencySpecialist()
        }

    def rank_actions(self,
                     potential_actions: List[Dict], # List of edge data dicts
                     context: StrategicContext,
                     target_nodes_data: List[Dict], # Corresponding target node data
                     graph: Optional["StrategicGraphMemory"] = None,
                     saturation_config: Optional[Dict[str, Any]] = None,
                    ):
        """
        Ranks actions based on the Multi-Objective Score.

        Args:
            potential_actions: edge-data dicts to rank.
            context: active StrategicContext (lens weights).
            target_nodes_data: node-data dict per action (parallel list).
            graph: optional StrategicGraphMemory — required for saturation detection.
                   If None, saturation is silently disabled (backward compatible).
            saturation_config: optional dict with thresholds. If None, saturation is disabled.

        Returns:
            (ranked_results, saturation_info)
              ranked_results: List of (ActionDict, TotalScore, BreakdownDict) sorted by Score desc.
              saturation_info: dict describing the saturation state of the active lens.

        Backward compatibility note:
            Older call sites unpacking only the ranked list will continue to work because
            this method always returns a 2-tuple; old code using `for a, s, b in result:`
            would break, but no such pattern exists in the current codebase. Callers that
            need only the ranked list can do `ranked, _ = ranker.rank_actions(...)`.
        """
        scored_actions = []

        for i, action in enumerate(potential_actions):
            node_data = target_nodes_data[i]

            # 1. Calculate Individual Scores
            scores = {}
            for key, specialist in self.specs.items():
                scores[key] = specialist.evaluate(action, node_data)

            # 2. Apply Lens Weights
            total_score = 0.0
            for key, weight in context.weights.items():
                total_score += scores.get(key, 0.0) * weight

            scored_actions.append((action, total_score, scores))

        # 3. Sort by Total Score
        scored_actions.sort(key=lambda x: x[1], reverse=True)

        # 4. Saturation detection (advisory; never alters the ranking).
        saturation_info = dict(_SATURATION_DISABLED)
        if graph is not None and saturation_config is not None and saturation_config.get("saturation_enabled", False):
            try:
                top_target = None
                if scored_actions:
                    # Best-ranked action's target node data — used for Explorer per-target percentile
                    top_target = target_nodes_data[
                        potential_actions.index(scored_actions[0][0])
                    ]
                saturation_info = self._detect_saturation(
                    context=context,
                    graph=graph,
                    config=saturation_config,
                    top_target_node_data=top_target,
                )
            except Exception as e:
                # Saturation detection must never break ranking.
                saturation_info = dict(_SATURATION_DISABLED)
                saturation_info["lens"] = "ERROR"
                saturation_info["metric_name"] = str(e)[:120]

        return scored_actions, saturation_info

    @staticmethod
    def _identify_active_lens(context: "StrategicContext") -> str:
        """Return canonical lens name from context weights, with deterministic tie-break."""
        weights = context.weights or {}
        if not weights:
            return "HAWK"
        max_w = max(weights.values())
        # Find all keys tied for the max
        tied = [k for k, v in weights.items() if v == max_w]
        for key in _LENS_KEY_PRIORITY:
            if key in tied:
                return _LENS_KEY_TO_NAME.get(key, "HAWK")
        return "HAWK"

    def _detect_saturation(
        self,
        context: "StrategicContext",
        graph: "StrategicGraphMemory",
        config: Dict[str, Any],
        top_target_node_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Per-lens saturation detector. Returns saturation_info dict."""
        info = dict(_SATURATION_DISABLED)
        info["lens"] = self._identify_active_lens(context)

        # Cold-graph guard
        try:
            n_nodes = graph.graph.number_of_nodes()
        except Exception:
            n_nodes = 0
        info["n_graph_nodes"] = int(n_nodes)
        min_nodes = int(config.get("saturation_min_graph_nodes", 10))
        if n_nodes < min_nodes:
            return info

        recent_window = int(config.get("saturation_recent_window", 10))
        metrics = graph.get_lens_signal_metric(info["lens"], recent_window)

        if info["lens"] == "EXPLORER":
            info["metric_name"] = "novelty_percentile"
            info["threshold"] = float(config.get("explorer_novelty_percentile", 0.75))
            # Prefer per-target percentile if we have a top-ranked target
            if top_target_node_data is not None:
                target_visits = int(top_target_node_data.get("visits", 0) or 0)
                value = graph.get_graph_novelty_percentile(target_visits)
            else:
                value = metrics.get("novelty_percentile", 0.0)
            info["metric_value"] = float(value)
            info["is_saturated"] = bool(value > info["threshold"])

        elif info["lens"] == "HAWK":
            info["metric_name"] = "accuracy_slope"
            info["threshold"] = float(config.get("hawk_accuracy_slope_threshold", 0.001))
            slope = metrics.get("accuracy_slope", 0.0)
            info["metric_value"] = float(slope)
            n_recent = metrics.get("n_recent_edges", 0)
            # Plateau when |slope| below threshold AND we have enough recent samples
            info["is_saturated"] = bool(n_recent >= recent_window and abs(slope) < info["threshold"])

        elif info["lens"] == "SAGE":
            info["metric_name"] = "stability_std"
            info["threshold"] = float(config.get("sage_stability_std_threshold", 0.01))
            std = metrics.get("stability_std", 0.0)
            info["metric_value"] = float(std)
            n_recent = metrics.get("n_recent_edges", 0)
            info["is_saturated"] = bool(n_recent >= recent_window and std < info["threshold"])

        elif info["lens"] == "ENGINEER":
            info["metric_name"] = "efficiency_coverage"
            info["threshold"] = float(config.get("engineer_coverage_threshold", 0.8))
            cov = metrics.get("efficiency_coverage", 1.0)
            info["metric_value"] = float(cov)
            info["is_saturated"] = bool(cov < info["threshold"])

        return info
