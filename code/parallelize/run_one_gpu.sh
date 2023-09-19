#!/bin/bash -x

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1            
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1  
#SBATCH --mem=0
#SBATCH --cpus-per-task=96
#SBATCH --time=04:00:00
#SBATCH --partition=booster
#SBATCH --account=training2324
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --reservation=ai_sc_day2

# To get number of cpu per task
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# activate env
source $HOME/course/$USER/sc_venv_template/activate.sh

# run script from above
start=$(date +%s)
srun python3 gpu_training.py
ELAPSED=$(($(date +%s) - start))

printf "elapsed: %s\n\n" "$(date -d@$ELAPSED -u +%H\ hours\ %M\ min\ %S\ sec)"