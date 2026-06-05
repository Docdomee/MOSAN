#!/bin/bash
# ============================================================================
# vLLM SETUP & MODEL DOWNLOAD SCRIPT (Gemma 4 31B Dense)
# ============================================================================
# Run this script ONCE on a login node (with internet access).
# Usage: bash scripts/setup_vllm.sh
# Requires HF_TOKEN for gated Gemma models: export HF_TOKEN=hf_...

echo "========================================="
echo "Setting up vLLM Environment & Downloading Gemma 4 31B Dense"
echo "========================================="

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

# 1. Create Dedicated vLLM Environment (fresh)
if conda info --envs | grep -q "vllm_env"; then
    echo "Removing existing 'vllm_env'..."
    conda env remove -n vllm_env -y
fi
echo "Creating 'vllm_env'..."
conda create -n vllm_env python=3.10 -y

conda activate vllm_env

# 2. Install torch first (must precede vLLM to avoid ABI mismatch)
# Tested stack: vllm==0.19.0 + torch==2.6.0+cu124 on A100 (driver 570, CUDA 12.8)
# NOTE: vllm>=0.20.x bundles CUDA 13 runtime — incompatible with driver<=570 (max CUDA 12.8)
echo "Installing torch..."
pip install "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124

# 3. Install vLLM 0.19.0 — first version with Gemma 4 support, uses CUDA 12.x
echo "Installing vLLM..."
pip install "vllm==0.19.0" --extra-index-url https://download.pytorch.org/whl/cu124

# 4. Install transformers from source (required for gemma4 architecture support)
echo "Installing transformers from source..."
pip install git+https://github.com/huggingface/transformers.git

# 5. Install huggingface_hub for model download
pip install huggingface_hub

# 6. Verify stack
echo "Verifying installation..."
python -c "
import torch, torchvision, vllm, transformers
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('vllm:', vllm.__version__)
print('transformers:', transformers.__version__)
print('Stack OK')
"

# 7. Download Model to Cache
# Target Model: Gemma 4 31B Dense (BF16)
# VRAM: ~62GB BF16 → fits on 4x A100 40GB (TP=4, ~15.5GB/GPU, ~24GB KV cache headroom)
# Gated model — HF_TOKEN required: export HF_TOKEN=hf_...
MODEL_ID="google/gemma-4-31b-it"

# Base cache dir
CACHE_BASE="$(pwd)/processed_data/model_cache"
# Specific model dir (must match run_unified_short_job.sh logic)
TARGET_DIR="$CACHE_BASE/gemma-4-31b-it"

if [ -d "$TARGET_DIR" ] && [ -f "$TARGET_DIR/config.json" ]; then
    echo "Model already exists at $TARGET_DIR. Skipping download."
else
    echo "Downloading model $MODEL_ID to $TARGET_DIR..."
    mkdir -p "$TARGET_DIR"
    HUGGING_FACE_HUB_TOKEN=$HF_TOKEN huggingface-cli download \
        $MODEL_ID \
        --local-dir "$TARGET_DIR" \
        --local-dir-use-symlinks False
fi

echo "========================================="
echo "Setup Complete!"
echo "vLLM Environment: vllm_env"
echo "Model path: $TARGET_DIR"
echo "========================================="
