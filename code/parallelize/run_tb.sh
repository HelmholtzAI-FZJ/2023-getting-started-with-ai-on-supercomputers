#!/bin/bash -x

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=4             
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1  
#SBATCH --mem=0
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00
#SBATCH --partition=booster
#SBATCH --account=training2321
#SBATCH --output=%j.out
#SBATCH --error=%j.err

CUDA_VISIBLE_DEVICES=0,1,2,3

# activate env
source ../sc_venv_template/activate.sh

# run script from above
start=$(date +%s)
srun python3 tb.py
ELAPSED=$(($(date +%s) - start))

printf "elapsed: %s\n\n" "$(date -d@$ELAPSED -u +%H\ hours\ %M\ min\ %S\ sec)"