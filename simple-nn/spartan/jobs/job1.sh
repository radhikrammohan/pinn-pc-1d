#!/bin/bash
#SBATCH --job-name=1d-pinn
#SBATCH --output=output-logs/output_%j.txt
#SBATCH --error=output-logs/error_%j.txt
#SBATCH --time=23:55:00
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --account=OD-217410

# Define the timestamp format
TIMESTAMP=$(date +%Y-%m-%d_%H-%M)

module load python
source pinn2/bin/activate

echo " settings_peter_geo+mat simulation with job id $SLURM_JOB_ID"
python main_peter_geo_mat_3.py


