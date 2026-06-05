#!/bin/bash
#SBATCH --job-name=agent_quick_test
#SBATCH --output=logs/quick_test_%j.out
#SBATCH --error=logs/quick_test_%j.err

# ============================================================================
# Quick Agent Test - Run main.py for 3 steps
# ============================================================================
# This script runs the actual agent system for a few steps to verify:
# - Environment setup works
# - Agent initialization works
# - Tool execution works
# - No import/path errors
# ============================================================================

# ============================================================================
# GPU CONFIGURATION
# ============================================================================
#SBATCH --gres=gpu:3                       # 1 GPU is enough for quick test
#SBATCH --partition=short                 # Fast queue

# ============================================================================
# COMPUTE RESOURCES
# ============================================================================
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================================================
# TIME LIMITS
# ============================================================================
#SBATCH --time=06:00:00                    # 30 minutes should be plenty

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "========================================="
echo "Quick Agent Test (3 steps)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================="
echo ""

# Create logs directory
mkdir -p logs

# ============================================================================
# PYTHON ENVIRONMENT
# ============================================================================

# Initialize Conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate agent_gpu_env

# Verify
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

# ============================================================================
# VERIFY GPU ACCESS
# ============================================================================
echo "Checking GPU..."
nvidia-smi
echo ""
python -c "import tensorflow as tf; print(f'TensorFlow GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"
echo ""

# ============================================================================
# NAVIGATE TO PROJECT
# ============================================================================
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"
echo ""

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# ============================================================================
# RUN AGENT FOR 3 STEPS
# ============================================================================
echo "========================================="
echo "Starting Agent (50 steps)"
echo "========================================="
echo ""

# Override num_steps to 3 via Hydra command line
# This will run the agent for only 3 optimization steps
python cluster/main.py execution.num_steps=250

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Test Results"
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Agent ran successfully for 3 steps!"
    echo ""
    echo "Check the logs for details:"
    echo "  - Full output: logs/quick_test_${SLURM_JOB_ID}.out"
    echo "  - Errors: logs/quick_test_${SLURM_JOB_ID}.err"
    echo "  - Results: cluster/processed_data/logs/results_log.json"
else
    echo "✗ Agent failed with exit code: $EXIT_CODE"
    echo ""
    echo "Check error log: logs/quick_test_${SLURM_JOB_ID}.err"
fi
echo "========================================="

echo ""
echo "Job completed at: $(date)"
