#!/bin/bash -x

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4    # This needs to match Trainer(devices=...)
#SBATCH --mem=0
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --partition=booster
#SBATCH --account=training2321
#SBATCH --output=%j.out
#SBATCH --error=%j.err


# activate env
source ../sc_venv_template/activate.sh

# run script from above
start=$(date +%s)
srun python3 ddp.py
ELAPSED=$(($(date +%s) - start))

printf "elapsed: %s\n\n" "$(date -d@$ELAPSED -u +%H\ hours\ %M\ min\ %S\ sec)"