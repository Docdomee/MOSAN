import os
import sys

# --- FRAMEWORK STABILITY FIX (Entry Point) ---
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ---------------------------------------------

from celery import Celery
from celery.schedules import crontab

# --- CRITICAL FIX: Explicit GPU Initialization ---
# Ensure TensorFlow memory growth is set before ANY other import triggers it.
# This prevents race conditions with vLLM when Celery starts.
try:
    # Try importing from root directly (if running from root)
    from system_utils import setup_gpu_memory
    setup_gpu_memory()
except ImportError:
    # If running from cluster/ or script folder, ensure root is in path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)
    try:
        from system_utils import setup_gpu_memory
        setup_gpu_memory()
    except ImportError as e:
        print(f"Warning: Could not import system_utils for GPU setup: {e}")
# -------------------------------------------------
from celery.schedules import crontab
import os

# Define the RabbitMQ/Redis Broker URL
# Default: localhost (headless/dev mode). Docker workers override via env var:
#   CELERY_BROKER_URL=redis://host.docker.internal:16379/0
DEFAULT_BROKER_URL = 'redis://host.docker.internal:16379/0'
DEFAULT_BACKEND_URL = 'redis://host.docker.internal:16379/0'

broker_url = os.environ.get('CELERY_BROKER_URL', DEFAULT_BROKER_URL)
backend_url = os.environ.get('CELERY_RESULT_BACKEND', DEFAULT_BACKEND_URL)

print(f"[Celery] Connection Config: Broker={broker_url}, Backend={backend_url}")

app = Celery('cluster_tasks',
             broker=broker_url,
             backend=backend_url,
             include=['cluster.tasks', 'cluster.cognitive_tasks'])

# Alias expected by workers started with --app cluster.celery_app
celery_app = app

# Configuration
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # Worker settings - preventing memory leaks
    worker_max_tasks_per_child=10, 
    # Queue settings can be added here
    beat_schedule={
        'monitor-resources-every-hour': {
            'task': 'cluster.tasks.monitor_resources_task',
            'schedule': crontab(minute=0, hour='*'), # Every hour
        },
        # Graph maintenance moved to step-based trigger in main.py
        # (respects strategic_advisor.enabled and maintenance_interval from config.yaml)
        'daily-report-midnight': {
            'task': 'cluster.cognitive_tasks.run_daily_report_task',
            'schedule': crontab(minute=0, hour=0), # Daily at midnight
        },
    }
)

if __name__ == '__main__':
    app.start()
