import math
import numpy as np
import json
from rl_utils import ALL_MODEL_ARCHITECTURES

class MetricSpecialist:
    """
    Base class for MOSAN Specialists.
    Each specialist evaluates a proposed edge (transition) based on its specific objective.
    """
    
    def evaluate(self, edge_data: dict, target_node_data: dict) -> float:
        """
        Returns a score (typically 0.0 to 1.0, or unbounded positive/negative)
        representing how desirable this transition is for this specialist.
        
        Args:
            edge_data: Dictionary containing 'impact_vector' and 'avg_reward'.
            target_node_data: Dictionary containing 'vector' and 'visit_count'.
            
        Impact Vector Schema (V2.1):
        [0]: Delta Accuracy (Positive Good)
        [1]: Delta Overfitting (Negative Good - i.e., reducing overfitting)
        [2]: Delta Training Speed (Positive Good - i.e. faster)
        [3]: Delta Inference Latency (Negative Good - i.e. lower latency)
        [4]: Delta Memory (Negative Good - i.e. lower memory)
        """
        raise NotImplementedError

    def _get_vector(self, edge_data: dict) -> list:
        """Helper to get Q-Vector (Long Term) or Impact Vector (Immediate)."""
        q_vec = edge_data.get("q_vector")
        if q_vec and isinstance(q_vec, list) and len(q_vec) > 0:
            return q_vec
        return edge_data.get("impact_vector", [0]*5)

class AccuracySpecialist(MetricSpecialist):
    """
    THE HAWK 🦅
    Focus: Maximizing Validation Accuracy.
    """
    def evaluate(self, edge_data: dict, target_node_data: dict) -> float:
        vector = self._get_vector(edge_data)
        # Delta Accuracy from impact vector (index 0)
        delta_acc = vector[0] if len(vector) > 0 else 0.0
        
        # Determine index of best_arch_accuracy in state vector
        # State = [arch_one_hot (M), perf_vector (5), tool_one_hot (T)]
        # Acc is at index M + 0
        ACC_INDEX = len(ALL_MODEL_ARCHITECTURES)
        
        target_vec = target_node_data.get("vector", [])
        if isinstance(target_vec, str):
            target_vec = json.loads(target_vec)
            
        abs_acc = 0.0
        if len(target_vec) > ACC_INDEX:
             abs_acc = target_vec[ACC_INDEX]
             
        # Combined Score: Historic Gain (Impact/Q) + State Value (Absolute Acc)
        score = delta_acc + abs_acc
             
        return score

class StabilitySpecialist(MetricSpecialist):
    """
    THE SAGE 🦉
    Focus: Minimizing Overfitting and ensuring stability.
    """
    def evaluate(self, edge_data: dict, target_node_data: dict) -> float:
        vector = self._get_vector(edge_data)
        # Overfitting impact is index 1. Negative Good.
        # Specialist wants to MAXIMIZE score.
        delta_overfit = vector[1] if len(vector) > 1 else 0.0
        
        # Stability index in state vector: perf_vector[3]
        # Index = M (arch) + 3
        STAB_INDEX = len(ALL_MODEL_ARCHITECTURES) + 3
        
        target_vec = target_node_data.get("vector", [])
        if isinstance(target_vec, str):
            target_vec = json.loads(target_vec)
            
        abs_stab = 0.0
        if len(target_vec) > STAB_INDEX:
            abs_stab = target_vec[STAB_INDEX]
            
        # Score = (Reduced Overfitting) + (High Stability)
        # Note: stability is usually std_dev (lower is better), so we use (1.0 - stability)
        score = (-delta_overfit) + (1.0 - abs_stab)
        
        return score

class NoveltySpecialist(MetricSpecialist):
    """
    THE EXPLORER 🧭
    Focus: Discovering new states.
    """
    def evaluate(self, edge_data: dict, target_node_data: dict) -> float:
        # Novelty is structural, not learned via Q-values yet.
        # 1. Information Gain: Inverse of visit count.
        visits = target_node_data.get("visits", 0)
        
        if visits == 0:
             return 1.0 # Maximum novelty
        
        # Decay function: 1 / sqrt(visits)
        score = 1.0 / math.sqrt(visits)
        
        return score

class EfficiencySpecialist(MetricSpecialist):
    """
    THE ENGINEER ⚙️
    Focus: Minimizing Latency and Memory.
    """
    def evaluate(self, edge_data: dict, target_node_data: dict) -> float:
        vector = self._get_vector(edge_data)
        # Handle shorter vectors (pad with 0)
        if len(vector) < 5:
            vector = vector + [0.0] * (5 - len(vector))
        
        delta_latency = vector[3]
        delta_memory = vector[4]
        
        # Simple heuristic: - (DeltaLat/10ms + DeltaMem/100MB)
        score = - (delta_latency / 10.0) - (delta_memory / 100.0)
        
        return score
