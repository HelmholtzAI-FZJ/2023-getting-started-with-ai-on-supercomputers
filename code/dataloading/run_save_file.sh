#!/bin/bash -x

#SBATCH --nodes=1          
#SBATCH --gres=gpu:1     
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=96
#SBATCH --time=10:00:00
#SBATCH --partition=booster
#SBATCH --account=training2402
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --reservation=training-booster-2024-03-13

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
source $HOME/course/$USER/sc_venv_template/activate.sh

time srun python save_imagenet_files.py  --dset_type "h5" --target_folder "/p/scratch/training2402/data/"