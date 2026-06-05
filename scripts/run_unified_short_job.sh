#!/bin/bash
#SBATCH --job-name=unified_mistral
#SBATCH --partition=long
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --time=3-06:00:00
#SBATCH --output=logs/unified_%j.out
#SBATCH --error=logs/unified_%j.err

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
export PYTHONUNBUFFERED=1 

# --- FRAMEWORK STABILITY ---
# Force pure-python protobuf to avoid 'MessageFactory' attribute errors on DGX
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# Disable oneDNN to prevent registration crashes
export TF_ENABLE_ONEDNN_OPTS=0
# ---------------------------
# NAVIGATE TO PROJECT ROOT
# Goal: Be where 'agent.py' and the 'cluster' package are located.

# 1. Start from SLURM_SUBMIT_DIR if set (Standard Slurm behavior)
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd $SLURM_SUBMIT_DIR
fi

# 2. Check where we are relative to agent.py
if [ -f "agent.py" ]; then
    echo "✅ Detected already in Repo Root (agent.py found)."
    # Do nothing, we are in the right place
elif [ -d "cluster" ] && [ -f "cluster/agent.py" ]; then
    echo "📂 Detected running from Parent Directory. Entering 'cluster'..."
    cd cluster
elif [ -f "../agent.py" ]; then
    echo "📂 Detected running from 'scripts' or subdirectory. Moving up..."
    cd ..
else
    echo "⚠️  WARNING: Could not auto-detect Root (agent.py not found). Current PWD: $(pwd)"
fi

echo "Working Directory: $(pwd)"
export MLFLOW_TRACKING_URI="file://$(pwd)/processed_data/mlruns"
mkdir -p logs

echo "========================================="
echo "Unified Agent Job"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================="

# 3. CONDA ACTIVATION (Robust)
CONDA_PATH="$HOME/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
else
    echo "⚠️  Conda script not found at $CONDA_PATH. Assuming 'conda' is in PATH."
fi

# Ensure env exists or fail
if conda env list | grep -q "agent_gpu_env"; then
    echo "✅ Activating conda environment: agent_gpu_env"
    conda activate agent_gpu_env
else
    echo "❌ ERROR: Conda environment 'agent_gpu_env' not found!"
    exit 1
fi

# Expose nvidia-* CUDA libraries installed via pip (cudnn, cublas, etc.)
# These .so files live under site-packages/nvidia/*/lib but are NOT on LD_LIBRARY_PATH by default,
# causing TensorFlow to fail with "Cannot dlopen some GPU libraries".
NVIDIA_LIBS=$(python -c "
import site, os
paths = []
for sp in site.getsitepackages():
    nv = os.path.join(sp, 'nvidia')
    if os.path.isdir(nv):
        for pkg in os.listdir(nv):
            lib = os.path.join(nv, pkg, 'lib')
            if os.path.isdir(lib):
                paths.append(lib)
print(':'.join(paths))
" 2>/dev/null)
if [ -n "$NVIDIA_LIBS" ]; then
    export LD_LIBRARY_PATH=$NVIDIA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    echo "✅ CUDA libs exposed: $NVIDIA_LIBS"
else
    echo "⚠️  No nvidia pip packages found — LD_LIBRARY_PATH unchanged."
fi

# ============================================================================
# 4. CONFIG CHECK (The Bible)
# ============================================================================
# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "❌ ERROR: config.yaml not found!"
    exit 1
fi

# Read Provider from config.yaml using python for reliability
# We default to 'Openrouter' if lookup fails to be safe avoiding vllm launch
PROVIDER=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['api'].get('provider', 'Openrouter'))" 2>/dev/null)

echo "📜 Config Source of Truth: Provider = $PROVIDER"

USE_VLLM=false
if [[ "$PROVIDER" == "vllm" ]]; then
    USE_VLLM=true
    echo "✅ Mode: LOCAL vLLM (Requested by Config)"
else
    echo "✅ Mode: EXTERNAL API (Config: $PROVIDER). Skipping vLLM."
fi


# ============================================================================
# CLEANUP FUNCTION (ROBUST)
# ============================================================================
cleanup() {
    echo "🛑 Rilevato segnale di stop/uscita. Pulizia in corso..."
    
    # 1. Spegnimento vLLM (se attivo)
    if [ "$USE_VLLM" = true ] && [ -n "$VLLM_PID" ]; then
        echo "Killing vLLM ($VLLM_PID)..."
        kill -SIGTERM $VLLM_PID 2>/dev/null
    fi

    # 2. Spegnimento Celery Worker (Graceful -> Warm Shutdown)
    if [ -n "$CELERY_WORKER_PID" ]; then
        echo "Stopping Celery Worker ($CELERY_WORKER_PID)..."
        kill -SIGTERM $CELERY_WORKER_PID 2>/dev/null
        # Opzionale: attendi qualche secondo che finisca il task
        # sleep 5 
    fi

    # 3. Spegnimento Celery Beat e Redis
    [ -n "$CELERY_BEAT_PID" ] && kill $CELERY_BEAT_PID 2>/dev/null
    [ -n "$REDIS_PID" ] && kill $REDIS_PID 2>/dev/null

    # 4. Sicurezza: uccidi eventuali figli rimasti orfani
    pkill -P $$ 2>/dev/null
    
    echo "✅ Pulizia completata."
}

# AGGIUNGI SIGTERM E SIGINT QUI SOTTO
trap cleanup EXIT SIGTERM SIGINT

# ============================================================================
# 5. PORT SELECTION (Dynamic)
# ============================================================================
get_free_port() {
    python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

REDIS_PORT=$(get_free_port)
# Only need VLLM port if using VLLM
if [ "$USE_VLLM" = true ]; then
    VLLM_PORT=$(get_free_port)
    while [ "$VLLM_PORT" == "$REDIS_PORT" ]; do
        VLLM_PORT=$(get_free_port)
    done
    echo "Selected Ports -> Redis: $REDIS_PORT | vLLM: $VLLM_PORT"
else
    echo "Selected Ports -> Redis: $REDIS_PORT"
fi


# Export Celery Config
export CELERY_BROKER_URL="redis://localhost:$REDIS_PORT/0"
export CELERY_RESULT_BACKEND="redis://localhost:$REDIS_PORT/0"


# ============================================================================
# 6. SMART GPU SELECTION & ALLOCATION
# ============================================================================
echo "Scanning for GPUs (Greedy Memory Sort)..."

# --- GPU SCAN INLINE (No file write to avoid Disk Quota errors) ---
PYTHON_OUTPUT=$(python -c "
import subprocess, os
def get_best_gpus():
    try:
        res = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'], encoding='utf-8')
        gpu_stats = []
        for line in res.strip().split('\n'):
            if not line: continue
            idx, mem = line.split(',')
            gpu_stats.append((int(mem.strip()), int(idx.strip())))
        gpu_stats.sort()
        selected = [str(idx) for mem, idx in gpu_stats if mem < 9000]
        return ','.join(selected)
    except: return ''

gpus = get_best_gpus()
if gpus: print(f'SELECTED_GPUS={gpus}')
else: print('NO_GPUS_AVAILABLE')
")

CLEAN_GPUS=$(echo "$PYTHON_OUTPUT" | grep "SELECTED_GPUS" | cut -d'=' -f2)
# -------------------------------------------------------------------

IFS=',' read -r -a GPU_ARRAY <<< "$CLEAN_GPUS"
NUM_FOUND=${#GPU_ARRAY[@]}

# --- ALLOCATION STRATEGY ---
# TP=2 (2x A100 40GB): ~16.5GB weights/GPU → only ~177MB free for KV cache → OOM (job 31393).
# TP=3 (3x A100 40GB): Fails because model dimensions (8192) are not divisible by 3 (ValidationError).
# TP=4 (4x A100 40GB): ~8.25GB weights/GPU → ~31GB free/GPU → Best stability and cache headroom.
if [ "$USE_VLLM" = true ]; then
    # Need at least 5 GPUs: 4 for vLLM (TP=4) + 1 for Agent
    if [ $NUM_FOUND -lt 5 ]; then
        echo "ERROR: vLLM Mode requires 5+ GPUs (4 vLLM TP=4, 1+ Agent). Found $NUM_FOUND."
        exit 1
    else
        # Give first 4 GPUs to vLLM with TP_SIZE=4, rest to Agent/Worker
        VLLM_GPUS="${GPU_ARRAY[0]},${GPU_ARRAY[1]},${GPU_ARRAY[2]},${GPU_ARRAY[3]}"
        AGENT_GPUS=$(IFS=,; echo "${GPU_ARRAY[*]:4}")
        TP_SIZE=4
    fi
    echo "Allocation (vLLM Mode, TP=$TP_SIZE): vLLM=$VLLM_GPUS | Agent=$AGENT_GPUS"
else
    # EXTERNAL API MODE -> GIVE IT ALL TO AGENT/WORKER
    if [ -z "$CLEAN_GPUS" ] || [ $NUM_FOUND -lt 1 ]; then
        echo "ERROR: Need at least 1 GPU for Agent/Worker. Found 0 free GPUs (all above 9000 MiB threshold)."
        exit 1
    fi
    AGENT_GPUS="$CLEAN_GPUS"
    echo "Allocation (External API Mode): Agent=$AGENT_GPUS"
fi


# ============================================================================
# 7. START REDIS (Database)
# ============================================================================
echo "Starting Redis on port $REDIS_PORT..."
redis-server --port $REDIS_PORT --bind 127.0.0.1 > logs/redis_${SLURM_JOB_ID}.log 2>&1 &
REDIS_PID=$!
echo "Redis managed [PID: $REDIS_PID]"
sleep 2

# PYTHONPATH must come first so 'cluster' resolves to our local package,
# not the system 'cluster' module that causes "no attribute 'celery_app'" errors.
export PYTHONPATH=$(pwd):$PYTHONPATH

# ============================================================================
# 8. START vLLM FIRST (ONLY IF ENABLED) — must be ready before workers start
# ============================================================================
if [ "$USE_VLLM" = true ]; then
    if conda env list | grep -q "vllm_env"; then
        echo "Switching to vllm_env for Model Server..."
        conda activate vllm_env
    else
        echo "vllm_env not found, staying in agent_gpu_env (Hope vllm is installed here!)"
    fi

    echo "Starting vLLM Server on port $VLLM_PORT..."
    MODEL_REPO="google/gemma-4-31b-it"

    # Default to Child (Standard)
    LOCAL_MODEL_DIR="$(pwd)/processed_data/model_cache/gemma-4-31b-it"
    # Check Parent (User Scenario)
    PARENT_MODEL_DIR="$(dirname $(pwd))/processed_data/model_cache/gemma-4-31b-it"

    if [ -f "$PARENT_MODEL_DIR/config.json" ]; then
        LOCAL_MODEL_DIR="$PARENT_MODEL_DIR"
    elif [ -f "$LOCAL_MODEL_DIR/config.json" ]; then
        LOCAL_MODEL_DIR="$LOCAL_MODEL_DIR"
    fi
    echo "Using Model Dir: $LOCAL_MODEL_DIR"

    # Ensure model exists
    if [ ! -f "$LOCAL_MODEL_DIR/config.json" ]; then
        echo "⚠️ Model missing. Redownloading..."
        rm -rf $LOCAL_MODEL_DIR
        mkdir -p $LOCAL_MODEL_DIR
        HUGGING_FACE_HUB_TOKEN=$HF_TOKEN huggingface-cli download $MODEL_REPO --local-dir "$LOCAL_MODEL_DIR" --local-dir-use-symlinks False
    fi

    # Gemma 4 31B Dense — BF16 on 4x A100 40GB (TP=4)
    # ~15.5GB weights/GPU, leaving ~24GB free per GPU for KV cache.

    CUDA_VISIBLE_DEVICES=$VLLM_GPUS python -m vllm.entrypoints.openai.api_server \
        --model $LOCAL_MODEL_DIR \
        --trust-remote-code \
        --dtype bfloat16 \
        --tensor-parallel-size $TP_SIZE \
        --max-model-len 32768 \
        --max-num-batched-tokens 4096 \
        --gpu-memory-utilization 0.90 \
        --port $VLLM_PORT \
        > logs/vllm_${SLURM_JOB_ID}.log 2>&1 &

    VLLM_PID=$!
    echo "vLLM managed [PID: $VLLM_PID] (GPUs: $VLLM_GPUS)"

    # Wait for vLLM to be ready BEFORE starting workers.
    # Workers that start before vllm is ready will use 'local-model' as model name
    # because _rehydrate_agent() cannot yet query /v1/models (job 31393 regression).
    echo "Waiting for vLLM to be ready on port $VLLM_PORT..."
    MAX_RETRIES=120
    COUNTER=0
    while [ $COUNTER -lt $MAX_RETRIES ]; do
        if curl -s http://localhost:$VLLM_PORT/v1/models > /dev/null; then
            echo "vLLM is ready!"
            break
        fi
        sleep 10
        let COUNTER=COUNTER+1
    done
    if [ $COUNTER -eq $MAX_RETRIES ]; then
        echo "vLLM Timeout. Check logs."
        tail -n 20 logs/vllm_${SLURM_JOB_ID}.log
        exit 1
    fi
else
    echo "⚠️  Skipping vLLM Launch (Not requested in config)."
fi

# ============================================================================
# 9. START CELERY WORKER (Training Engine)
# Started AFTER vLLM is ready so _rehydrate_agent() detects the correct model name.
# ============================================================================
echo "Starting Celery Worker..."

# Switch back to agent env after vllm setup
conda activate agent_gpu_env

# Verify Celery is installed
if ! command -v celery &> /dev/null; then
    echo "❌ ERROR: Celery command not found!"
    exit 1
fi

# The Worker runs on AGENT_GPUS
export CUDA_VISIBLE_DEVICES=$AGENT_GPUS
if [ -z "$AGENT_GPUS" ]; then
    NUM_AGENT_GPUS=1
else
    NUM_AGENT_GPUS=$(echo $AGENT_GPUS | awk -F"," '{print NF}')
fi
if [ -z "$NUM_AGENT_GPUS" ] || [ "$NUM_AGENT_GPUS" -lt 1 ]; then NUM_AGENT_GPUS=1; fi

celery -A cluster.celery_app worker --loglevel=info --concurrency=$NUM_AGENT_GPUS \
    --hostname=worker@%h \
    > logs/celery_worker_${SLURM_JOB_ID}.log 2>&1 &
CELERY_WORKER_PID=$!
echo "Celery Worker managed [PID: $CELERY_WORKER_PID] (GPUs: $AGENT_GPUS, Concurrency: $NUM_AGENT_GPUS)"

# ============================================================================
# 10. START CELERY BEAT (Scheduler)
# Started after worker is up so beat tasks land on a ready worker.
# ============================================================================
echo "Starting Celery Beat..."
celery -A cluster.celery_app beat --loglevel=info \
    > logs/celery_beat_${SLURM_JOB_ID}.log 2>&1 &
CELERY_BEAT_PID=$!
echo "Celery Beat managed [PID: $CELERY_BEAT_PID]"

# ============================================================================
# 11. LAUNCH AGENT (Brain)
# ============================================================================
echo "Starting Agent..."
# Switch back to agent env if we switched
conda activate agent_gpu_env

if [ ! -f "main.py" ] && [ -f "cluster/main.py" ]; then
    SCRIPT_PATH="cluster/main.py"
else
    SCRIPT_PATH="main.py"
fi

if [ -f "$SCRIPT_PATH" ]; then
    if [ "$USE_VLLM" = true ]; then
        # OVERRIDE: Force vLLM using local port
        echo "Launching Agent with FORCED vLLM config..."
        CUDA_VISIBLE_DEVICES=$AGENT_GPUS python $SCRIPT_PATH \
            execution.num_steps=1000 \
            api.provider="vllm" \
            api.local_llm_url="http://localhost:$VLLM_PORT/v1"
    else
        # RESPECT CONFIG: Do not override provider
        echo "Launching Agent with DEFAULT config (from config.yaml)..."
        CUDA_VISIBLE_DEVICES=$AGENT_GPUS python $SCRIPT_PATH \
            execution.num_steps=1000
    fi
else
    echo "❌ ERROR: main.py not found!"
    exit 1
fi

EXIT_CODE=$?
exit $EXIT_CODE
