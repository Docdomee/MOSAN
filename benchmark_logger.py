import csv
import os
import time
from datetime import datetime
from state_manager import PERSISTENT_PATHS

BENCHMARK_FILE = os.path.join(PERSISTENT_PATHS["logs_dir"], "benchmark_results.csv")

def log_benchmark_result(
    experiment_id: str,
    method: str,  # 'Supernet' or 'RandomSearch'
    dataset: str,
    accuracy: float,
    duration_seconds: float,
    gpu_hours: float,
    inference_time_ms: float = 0.0,
    params_count: int = 0,
    model_size_mb: float = 0.0,
    params: dict = None
):
    """
    Logs a benchmark result to a CSV file for easy plotting.
    """
    file_exists = os.path.isfile(BENCHMARK_FILE)
    
    with open(BENCHMARK_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if new file
        if not file_exists:
            writer.writerow([
                "timestamp", 
                "experiment_id", 
                "method", 
                "dataset", 
                "accuracy", 
                "duration_seconds", 
                "gpu_hours", 
                "inference_time_ms",
                "params_count",
                "model_size_mb",
                "params_json"
            ])
            
        writer.writerow([
            datetime.now().isoformat(),
            experiment_id,
            method,
            dataset,
            f"{accuracy:.4f}",
            f"{duration_seconds:.2f}",
            f"{gpu_hours:.4f}",
            f"{inference_time_ms:.2f}",
            params_count,
            f"{model_size_mb:.2f}",
            str(params)
        ])
    
    print(f"[Benchmark] Result logged to {BENCHMARK_FILE}")
