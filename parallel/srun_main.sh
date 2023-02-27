#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --output=serial/output/%j.out
#SBATCH --error=serial/output/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --account=training2303

mkdir -p serial/output
source sc_venv_template/activate.sh

srun python -u main.py --data_dir "tiny-imagenet-200/" --log "serial/logs/" 