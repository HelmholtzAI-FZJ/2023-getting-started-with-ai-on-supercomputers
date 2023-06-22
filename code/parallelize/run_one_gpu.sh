#!/bin/bash -x

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1            
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1  
#SBATCH --mem=0
#SBATCH --cpus-per-task=96
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --account=training2321
#SBATCH --output=%j.out
#SBATCH --error=%j.err

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# activate env
source ../sc_venv_template/activate.sh

# run script from above
start=$(date +%s)
srun python3 ddp.py
ELAPSED=$(($(date +%s) - start))

printf "elapsed: %s\n\n" "$(date -d@$ELAPSED -u +%H\ hours\ %M\ min\ %S\ sec)"