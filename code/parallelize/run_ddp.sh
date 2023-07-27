#!/bin/bash -x

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=64             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4    # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --account=training2324
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --reservation=ai_sc_day2

export CUDA_VISIBLE_DEVICES=0,1,2,3

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

# activate env
source ../sc_venv_template/activate.sh

# run script from above
start=$(date +%s)
srun python3 ddp.py
ELAPSED=$(($(date +%s) - start))

printf "elapsed: %s\n\n" "$(date -d@$ELAPSED -u +%H\ hours\ %M\ min\ %S\ sec)"