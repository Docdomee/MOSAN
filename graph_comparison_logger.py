/# graph_comparison_logger.py
"""
Logger per confrontare performance dell'agente con e senza graph planner.
Permette A/B testing tra modalità graph-based e standard flow.
Integrato con MLflow per tracking e visualizzazione.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from ui_logger import log

# MLflow integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    log("[Graph Comparison] MLflow not available. Metrics will only be logged to JSON.")


class GraphComparisonLogger:
    """
    Logs performance metrics to compare graph-based vs standard planning.
    Integrates with MLflow for experiment tracking.
    """
    
    def __init__(
        self, 
        log_dir: str = "processed_data/logs/graph_comparison",
        mlflow_enabled: bool = True,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: str = "Graph_Planner_Comparison"
    ):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Current run metadata
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = os.path.join(
            self.log_dir, 
            f"comparison_{self.run_id}.json"
        )
        
        # MLflow setup
        self.mlflow_enabled = mlflow_enabled and MLFLOW_AVAILABLE
        self.mlflow_run = None
        
        if self.mlflow_enabled:
            try:
                # Set tracking URI
                if mlflow_tracking_uri:
                    mlflow.set_tracking_uri(mlflow_tracking_uri)
                else:
                    # Use default from project
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    mlflow.set_tracking_uri(f"file:///{os.path.join(project_root, 'mlruns')}")
                
                # Set or create experiment
                mlflow.set_experiment(mlflow_experiment_name)
                
                log(f"[Graph Comparison] MLflow tracking enabled at {mlflow.get_tracking_uri()}")
            except Exception as e:
                log(f"[Graph Comparison] MLflow setup failed: {e}. Continuing without MLflow.")
                self.mlflow_enabled = False
        
        # Metrics storage
        self.metrics = {
            "run_id": self.run_id,
            "start_time": datetime.now().isoformat(),
            "graph_enabled": None,  # Will be set on first log
            "steps": []
        }
    
    def start_mlflow_run(self, **run_tags):
        """
        Start an MLflow run for this comparison session.
        Tags might include: graph_enabled, architecture, dataset, etc.
        """
        if not self.mlflow_enabled:
            return
        
        try:
            # Start a parent run for the entire session
            self.mlflow_run = mlflow.start_run(
                run_name=f"Graph_Comparison_{self.run_id}",
                tags={
                    "run_id": self.run_id,
                    "comparison_type": "graph_vs_standard",
                    **run_tags
                }
            )
            log(f"[Graph Comparison] MLflow run started: {self.mlflow_run.info.run_id}")
        except Exception as e:
            log(f"[Graph Comparison] Failed to start MLflow run: {e}")
            self.mlflow_enabled = False
    
    def end_mlflow_run(self):
        """End the MLflow run and log final summary metrics."""
        if not self.mlflow_enabled or not self.mlflow_run:
            return
        
        try:
            # Log final summary metrics
            summary = self.get_summary()
            
            if "graph_mode_stats" in summary and summary["graph_mode_stats"].get("count", 0) > 0:
                mlflow.log_metrics({
                    "final_graph_avg_reward": summary["graph_mode_stats"]["avg_reward"],
                    "final_graph_success_rate": summary["graph_mode_stats"]["avg_success_rate"],
                    "final_graph_total_errors": summary["graph_mode_stats"]["total_errors"]
                })
            
            if "standard_mode_stats" in summary and summary["standard_mode_stats"].get("count", 0) > 0:
                mlflow.log_metrics({
                    "final_standard_avg_reward": summary["standard_mode_stats"]["avg_reward"],
                    "final_standard_success_rate": summary["standard_mode_stats"]["avg_success_rate"],
                    "final_standard_total_errors": summary["standard_mode_stats"]["total_errors"]
                })
            
            if "reward_improvement" in summary.get("comparison", {}):
                mlflow.log_metrics({
                    "final_reward_improvement_pct": summary["comparison"]["reward_improvement"]["improvement_pct"],
                    "final_success_improvement_pct": summary["comparison"]["success_rate_improvement"]["improvement_pct"]
                })
            
            # Log the JSON file as artifact
            mlflow.log_artifact(self.current_log_file, artifact_path="comparison_logs")
            
            mlflow.end_run()
            log("[Graph Comparison] MLflow run ended successfully")
        except Exception as e:
            log(f"[Graph Comparison] Error ending MLflow run: {e}")
    
    def log_step(
        self,
        step_number: int,
        graph_enabled: bool,
        planning_mode: str,  # "graph" | "standard" | "graph_fallback"
        plan_details: Dict[str, Any] = None,
        execution_results: List[Dict[str, Any]] = None,
        step_reward: float = 0.0,
        step_duration_seconds: float = 0.0
    ):
        """
        Log metrics for a single step.
        
        Args:
            step_number: Global step counter
            graph_enabled: Whether graph planner was enabled (from config)
            planning_mode: Which planning mode was used
            plan_details: Details about the plan (sequence, source, etc.)
            execution_results: Results from tool execution
            step_reward: Calculated reward for this step
            step_duration_seconds: Time taken for this step
        """
        # First log initialization
        if self.metrics["graph_enabled"] is None:
            self.metrics["graph_enabled"] = graph_enabled
            # Start MLflow run on first log
            if self.mlflow_enabled and not self.mlflow_run:
                self.start_mlflow_run(graph_enabled=str(graph_enabled))
        
        # Calculate success metrics
        total_tools = len(execution_results) if execution_results else 0
        successful_tools = sum(
            1 for r in (execution_results or []) 
            if r.get("status") == "completed"
        )
        error_count = total_tools - successful_tools
        
        step_data = {
            "step": step_number,
            "timestamp": datetime.now().isoformat(),
            "planning_mode": planning_mode,
            "graph_enabled": graph_enabled,
            "plan": {
                "sequence": plan_details.get("sequence", []) if plan_details else [],
                "source": plan_details.get("source", "unknown") if plan_details else "unknown",
                "expected_reward": plan_details.get("expected_reward", 0.0) if plan_details else 0.0,
                "context_match": plan_details.get("context_match", 0.0) if plan_details else 0.0
            },
            "execution": {
                "total_tools": total_tools,
                "successful_tools": successful_tools,
                "error_count": error_count,
                "success_rate": successful_tools / total_tools if total_tools > 0 else 0.0
            },
            "performance": {
                "step_reward": step_reward,
                "duration_seconds": step_duration_seconds
            }
        }
        
        self.metrics["steps"].append(step_data)
        
        # MLflow logging (per-step metrics)
        if self.mlflow_enabled and self.mlflow_run:
            try:
                # Log step metrics with prefixes based on planning mode
                prefix = planning_mode  # "graph", "standard", or "graph_fallback"
                
                mlflow.log_metrics({
                    f"{prefix}_reward": step_reward,
                    f"{prefix}_success_rate": step_data["execution"]["success_rate"],
                    f"{prefix}_error_count": error_count,
                    f"{prefix}_duration": step_duration_seconds,
                    f"{prefix}_tool_count": total_tools
                }, step=step_number)
                
                # Log plan details as params (for graph mode)
                if planning_mode == "graph" and plan_details:
                    mlflow.log_params({
                        f"step_{step_number}_source": plan_details.get("source", "unknown"),
                        f"step_{step_number}_context_match": plan_details.get("context_match", 0.0)
                    })
            except Exception as e:
                log(f"[Graph Comparison] MLflow logging failed for step {step_number}: {e}")
        
        # Save to disk after each step
        self._save()
        
        # Log summary
        log(
            f"[Graph Comparison] Step {step_number} | "
            f"Mode: {planning_mode} | "
            f"Success: {successful_tools}/{total_tools} | "
            f"Reward: {step_reward:.3f} | "
            f"Duration: {step_duration_seconds:.1f}s"
        )
    
    def _save(self):
        """Save metrics to disk."""
        try:
            with open(self.current_log_file, "w") as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            log(f"[Graph Comparison Logger] Error saving: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Calculate and return summary statistics.
        """
        if not self.metrics["steps"]:
            return {"error": "No steps logged yet"}
        
        # Separate graph vs standard steps
        graph_steps = [s for s in self.metrics["steps"] if s["planning_mode"] == "graph"]
        standard_steps = [s for s in self.metrics["steps"] if s["planning_mode"] == "standard"]
        fallback_steps = [s for s in self.metrics["steps"] if s["planning_mode"] == "graph_fallback"]
        
        def calc_stats(steps: List[Dict]) -> Dict[str, float]:
            if not steps:
                return {"count": 0}
            
            return {
                "count": len(steps),
                "avg_reward": sum(s["performance"]["step_reward"] for s in steps) / len(steps),
                "avg_success_rate": sum(s["execution"]["success_rate"] for s in steps) / len(steps),
                "avg_duration": sum(s["performance"]["duration_seconds"] for s in steps) / len(steps),
                "total_reward": sum(s["performance"]["step_reward"] for s in steps),
                "total_errors": sum(s["execution"]["error_count"] for s in steps)
            }
        
        return {
            "run_id": self.run_id,
            "total_steps": len(self.metrics["steps"]),
            "graph_enabled_in_config": self.metrics["graph_enabled"],
            "graph_mode_stats": calc_stats(graph_steps),
            "standard_mode_stats": calc_stats(standard_steps),
            "fallback_mode_stats": calc_stats(fallback_steps),
            "comparison": self._calculate_comparison(graph_steps, standard_steps)
        }
    
    def _calculate_comparison(
        self, 
        graph_steps: List[Dict], 
        standard_steps: List[Dict]
    ) -> Dict[str, Any]:
        """
        Calculate comparison metrics between graph and standard modes.
        """
        if not graph_steps or not standard_steps:
            return {"note": "Not enough data for comparison"}
        
        graph_avg_reward = sum(s["performance"]["step_reward"] for s in graph_steps) / len(graph_steps)
        standard_avg_reward = sum(s["performance"]["step_reward"] for s in standard_steps) / len(standard_steps)
        
        graph_avg_success = sum(s["execution"]["success_rate"] for s in graph_steps) / len(graph_steps)
        standard_avg_success = sum(s["execution"]["success_rate"] for s in standard_steps) / len(standard_steps)
        
        return {
            "reward_improvement": {
                "graph_avg": graph_avg_reward,
                "standard_avg": standard_avg_reward,
                "delta": graph_avg_reward - standard_avg_reward,
                "improvement_pct": ((graph_avg_reward - standard_avg_reward) / standard_avg_reward * 100) 
                    if standard_avg_reward != 0 else 0.0
            },
            "success_rate_improvement": {
                "graph_avg": graph_avg_success,
                "standard_avg": standard_avg_success,
                "delta": graph_avg_success - standard_avg_success,
                "improvement_pct": ((graph_avg_success - standard_avg_success) / standard_avg_success * 100)
                    if standard_avg_success != 0 else 0.0
            },
            "recommendation": self._get_recommendation(graph_avg_reward, standard_avg_reward)
        }
    
    def _get_recommendation(self, graph_reward: float, standard_reward: float) -> str:
        """
        Provide a recommendation based on performance.
        """
        improvement = ((graph_reward - standard_reward) / standard_reward * 100) if standard_reward != 0 else 0.0
        
        if improvement > 20:
            return "STRONG: Graph planner shows significant improvement. Keep enabled."
        elif improvement > 10:
            return "MODERATE: Graph planner shows improvement. Consider keeping enabled."
        elif improvement > 0:
            return "WEAK: Graph planner shows minor improvement. Monitor closely."
        elif improvement > -10:
            return "NEUTRAL: No significant difference. Either mode acceptable."
        else:
            return "NEGATIVE: Standard flow performing better. Consider disabling graph planner."
    
    def print_summary(self):
        """
        Print a formatted summary to console.
        """
        summary = self.get_summary()
        
        log("\n" + "="*70)
        log("GRAPH PLANNER COMPARISON SUMMARY")
        log("="*70)
        log(f"Run ID: {summary['run_id']}")
        log(f"Total Steps: {summary['total_steps']}")
        log(f"Graph Enabled in Config: {summary['graph_enabled_in_config']}")
        log("")
        
        # Graph mode stats
        if summary['graph_mode_stats']['count'] > 0:
            log("GRAPH MODE STATISTICS:")
            log(f"  Steps: {summary['graph_mode_stats']['count']}")
            log(f"  Avg Reward: {summary['graph_mode_stats']['avg_reward']:.3f}")
            log(f"  Avg Success Rate: {summary['graph_mode_stats']['avg_success_rate']:.2%}")
            log(f"  Total Errors: {summary['graph_mode_stats']['total_errors']}")
            log("")
        
        # Standard mode stats
        if summary['standard_mode_stats']['count'] > 0:
            log("STANDARD MODE STATISTICS:")
            log(f"  Steps: {summary['standard_mode_stats']['count']}")
            log(f"  Avg Reward: {summary['standard_mode_stats']['avg_reward']:.3f}")
            log(f"  Avg Success Rate: {summary['standard_mode_stats']['avg_success_rate']:.2%}")
            log(f"  Total Errors: {summary['standard_mode_stats']['total_errors']}")
            log("")
        
        # Comparison
        if "reward_improvement" in summary['comparison']:
            log("COMPARISON ANALYSIS:")
            log(f"  Reward Improvement: {summary['comparison']['reward_improvement']['improvement_pct']:.1f}%")
            log(f"  Success Rate Improvement: {summary['comparison']['success_rate_improvement']['improvement_pct']:.1f}%")
            log(f"  Recommendation: {summary['comparison']['recommendation']}")
        
        log("="*70 + "\n")


# Singleton instance
_logger_instance = None


def get_comparison_logger(
    mlflow_enabled: bool = True,
    mlflow_tracking_uri: Optional[str] = None,
    reset: bool = False
) -> GraphComparisonLogger:
    """
    Get the global comparison logger instance.
    
    Args:
        mlflow_enabled: Whether to enable MLflow tracking
        mlflow_tracking_uri: Custom MLflow tracking URI (optional)
        reset: If True, create a new instance (use for testing)
    """
    global _logger_instance
    
    if reset or _logger_instance is None:
        if _logger_instance is not None:
            # Cleanup previous instance
            _logger_instance.end_mlflow_run()
        
        _logger_instance = GraphComparisonLogger(
            mlflow_enabled=mlflow_enabled,
            mlflow_tracking_uri=mlflow_tracking_uri
        )
    
    return _logger_instance


def close_comparison_logger():
    """Cleanup function to properly close MLflow run."""
    global _logger_instance
    if _logger_instance is not None:
        _logger_instance.end_mlflow_run()
        _logger_instance = None
