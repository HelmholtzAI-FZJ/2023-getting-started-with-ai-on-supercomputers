#!/bin/bash -x

mkdir -p serial/output

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=02:00:00
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:1
#SBATCH --account=training2303

mkdir -p slurm_outputs

#SBATCH --output=slurm_outputs/out-%j.out
#SBATCH --error=slurm_outputs/err-%j.err


# Use this on tuesday
##SBATCH --reservation=training-20230229
# And this one on wednesday
##SBATCH --reservation=training-20230301

source sc_venv_template/activate.sh

srun python -u loader_timer.py