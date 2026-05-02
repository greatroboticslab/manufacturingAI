#!/bin/bash
#SBATCH --partition=research-gpu
#SBATCH --gres=gpu:2              
#SBATCH --time=2-00:00:00         
#SBATCH --job-name=xlstm_v4_ds
#SBATCH --output=xlstm_v4_ds.out
#SBATCH --error=xlstm_v4_ds.err
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8      

# ==================================================
# CONDA ENVIRONMENT
# ==================================================
source /opt/ohpc/pub/apps/miniconda/etc/profile.d/conda.sh
conda activate /projects/tc6d/my_deepspeed_env

# ==================================================
# ENVIRONMENT VARIABLES
# ==================================================
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

# ==================================================
# WORKING DIRECTORY
# ==================================================
cd /projects/tc6d/xLSTM_1_Stack

# ==================================================
# VERIFICATION
# ==================================================
echo "=============================================="
echo "Job Start: $(date)"
echo "Host: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo "Dir: $(pwd)"
echo "=============================================="

# ==================================================
# CUDA SETUP — set last so z00_lmod.sh can't wipe it
# ==================================================
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.1
export PATH=/opt/ohpc/pub/apps/cuda/12.1/bin:$PATH
export LD_LIBRARY_PATH=/opt/ohpc/pub/apps/cuda/12.1/lib64:$LD_LIBRARY_PATH
echo "CUDA_HOME: $CUDA_HOME"

# ==================================================
# RUN TRAINING
# ==================================================
deepspeed --num_gpus=2 xlstm_v4_deepspeed.py \
    --deepspeed \
    --deepspeed_config ds_config.json

# ==================================================
# COMPLETION
# ==================================================
echo "=============================================="
echo "Job End: $(date)"
echo "=============================================="