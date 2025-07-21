#!/bin/bash
#SBATCH --job-name=1d-pinn
#SBATCH --output=../output-logs/job_%j/output_%j.txt
#SBATCH --error=../output-logs/job_%j/error_%j.txt
#SBATCH --time=23:55:00
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100

# Define the timestamp format
TIMESTAMP=$(date +%Y-%m-%d_%H-%M)

module load python
source ../../../pinn-1d/bin/activate

export CUBLAS_WORKSPACE_CONFIG=:4096:8
python ../model_train_sal.py --job_id $SLURM_JOB_ID 


