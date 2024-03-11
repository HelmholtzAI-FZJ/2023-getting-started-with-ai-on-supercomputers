#!/bin/bash
#SBATCH --account=training2402
#SBATCH --mail-user=MYUSER@fz-juelich.de
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --job-name=matrix-multiplication
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=output.%j
#SBATCH --error=err.%j
#SBATCH --time=00:10:00
#SBATCH --partition=booster
##SBATCH --reservation=training-booster-2024-03-12 # For today only

cd /p/home/jusers/$USER/juwels/course/$USER
source sc_venv_template/activate.sh # Now we finally use the fastai module

srun python cats.py
