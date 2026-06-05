#!/bin/bash
# ============================================================================
# vLLM SETUP & MODEL DOWNLOAD SCRIPT
# ============================================================================
# Run this script ONCE on a login node (with internet access).

echo "========================================="
echo "Setting up vLLM Environment"
echo "========================================="

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

# 1. Create Dedicated vLLM Environment
# We use a separate env to avoid conflicts with TensorFlow in agent_gpu_env
if conda info --envs | grep -q "vllm_env"; then
    echo "Environment 'vllm_env' already exists. Skipping creation."
else
    echo "Creating 'vllm_env'..."
    conda create -n vllm_env python=3.10 -y
fi

conda activate vllm_env

# 2. Install vLLM
echo "Installing vLLM..."
pip install vllm huggingface_hub

# 3. Download Models to Cache
CACHE_DIR="$(pwd)/processed_data/model_cache"
mkdir -p $CACHE_DIR

# Download 72B (for 4-GPU runs)
MODEL_ID_72B="Qwen/Qwen2.5-72B-Instruct-AWQ"
echo "Downloading $MODEL_ID_72B..."
huggingface-cli download $MODEL_ID_72B --local-dir $CACHE_DIR/$MODEL_ID_72B --local-dir-use-symlinks False

# Download 32B (for 2-GPU runs)
MODEL_ID_32B="Qwen/Qwen2.5-32B-Instruct-AWQ"
echo "Downloading $MODEL_ID_32B..."
huggingface-cli download $MODEL_ID_32B --local-dir $CACHE_DIR/$MODEL_ID_32B --local-dir-use-symlinks False

echo "========================================="
echo "Setup Complete!"
echo "vLLM Environment: vllm_env"
echo "Model path: $CACHE_DIR/$MODEL_ID"
echo "========================================="
