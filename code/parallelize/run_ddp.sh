#!/bin/bash -x

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=16                        # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4               # When using pl it should always be set to 4
#SBATCH --mem=0
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --account=training2324
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --reservation=ai_sc_day2

export CUDA_VISIBLE_DEVICES=0,1,2,3       # Very important for multi node training

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# activate env
source $HOME/course/$USER/sc_venv_template/activate.sh

# run script from above
srun python3 ddp_training.py
