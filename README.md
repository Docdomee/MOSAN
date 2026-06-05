# MOSAN: an autonomous multi-objective agent for neural architecture discovery

An autonomous agent system designed to optimize, finetune, and manage deep learning models (1D CNN, Supernets, etc.) at scale using a **Single Ledger Architecture** (Analyst -> Theorist -> Decider).

## 🚀 Overview

This system orchestrates complex machine learning workflows across distributed clusters. It utilizes **vLLM** for reasoning, **Redis** for state management, and **Celery** for distributed task execution (the "muscle" of the operation).

---

## 🏗️ System Components

1.  **The Brain (Agent)**: Orchestrates reasoning, planning, and tool execution.
2.  **The Muscle (Celery Workers)**: Executes heavy training, finetuning, and evaluation tasks.
3.  **The Broker (Redis)**: Manages the communication and task queue between the Brain and the Muscle.
4.  **The Oracle (vLLM)**: Provides local LLM inference for high-throughput, private reasoning.
5.  **The Archivist (MLflow)**: Tracks every experiment, metric, and agent decision.

---

## 📦 Installation & Setup

### 1. Environment Setup
We recommend separate environments for the Agent and the Model Server (vLLM) to avoid dependency conflicts.

#### **Agent & Worker Environment**
```bash
conda create -n agent_gpu_env python=3.11 -y
conda activate agent_gpu_env
pip install -r requirements.txt
pip install "celery[redis]"
```

#### **vLLM Environment (Optional)**
Use the automated setup script for **Gemma 4 31B Dense** (or similar models):
```bash
export HF_TOKEN=your_token_here
bash scripts/setup_vllm.sh
```

---

## 📡 Services Configuration

### 1. Redis (The Heart of Communication)
Redis is mandatory for the agent to communicate with training workers.

*   **Docker (Recommended for Local/Servers)**:
    ```bash
    docker run -d --name agent-redis -p 6379:6379 redis:alpine
    ```
*   **Conda/Manual (Recommended for HPC/No-Root Clusters)**:
    ```bash
    conda install -c conda-forge redis-server
    # Start it in the background
    redis-server --port 6379 --bind 127.0.0.1 &
    ```

### 2. Celery Worker (The Training Engine)
Workers handle the training epochs and supernet evaluations. Ensure Redis is running first.
```bash
export CELERY_BROKER_URL="redis://localhost:6379/0"
# Start the worker (Concurrency should match number of available GPUs)
celery -A cluster.celery_app worker --loglevel=info --concurrency=1
```

---

## ⚙️ Configuration (`config.yaml`)

Copy the template and fill in your details:
```bash
cp config.example.yaml config.yaml
```

**Key Fields:**
- `api.provider`: Set to `"vllm"` for local inference or `"Openrouter"` for cloud.
- `api.local_llm_url`: Typically `http://localhost:8000/v1`.
- `memory.backend`: Use `"sqlite"` for robust strategy persistence.

---

## 🤖 Running the System

### Manual / Interactive Mode
```bash
python main.py
```

### Slurm Cluster Mode (Automated)
The system includes a **Unified Launch Script** that handles Redis, vLLM, and Celery setup automatically on a single node:
```bash
sbatch scripts/run_unified_short_job.sh
```

---

## 👁️ Monitoring

| Service | Log Location | Purpose |
| :--- | :--- | :--- |
| **Agent** | `logs/unified_*.out` | Reasoning, planning, and tool calls. |
| **Workers** | `logs/celery_worker_*.log`| Training progress, accuracy, and worker errors. |
| **vLLM** | `logs/vllm_*.log` | LLM server inference stats. |
| **MLflow** | `processed_data/mlruns` | Centralized experiment and metric tracking. |

To view the MLflow UI:
```bash
mlflow ui --port 5000
```

---

## 🐳 Docker Deployment
To build and run the complete agent environment:
```bash
docker build -t ai-architect-agent .
docker run --gpus all -v $(pwd)/processed_data:/app/processed_data ai-architect-agent
```

---

## 📜 Architecture Detail
The system follows a **Single Ledger** design. Every cycle, the **Analyst** scans MLflow for results, the **Theorist** formulates a training hypothesis, and the **Decider** triggers the Celery worker via Redis. This decoupling allows the agent to "think" about strategy while the GPUs are busy training.

---

## 📝 Citation / How to cite

If you use this software in academic work, please cite the accompanying MOSAN paper and/or the software citation described in `CITATION.cff`.
