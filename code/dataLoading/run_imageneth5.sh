#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --account=training2324
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --reservation=ai_sc_day2

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

source $HOME/course/$USER/sc_venv_template/activate.sh

time srun python imageNetH5.py -r "/p/scratch/training2324/data/ImageNet.h5"
