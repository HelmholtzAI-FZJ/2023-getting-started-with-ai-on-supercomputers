#!/bin/bash -x

#SBATCH --nodes=1            
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=96
#SBATCH --time=06:00:00
#SBATCH --partition=booster
#SBATCH --account=training2402
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --reservation=training-booster-2024-03-13

# To get number of cpu per task
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# activate env
source $HOME/course/$USER/sc_venv_template/activate.sh

# run script from above
time srun python3 gpu_training.py
