#!/bin/bash -x

#SBATCH --nodes=16                     # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4                   # Use the 4 GPUs available
#SBATCH --ntasks-per-node=4            # When using pl it should always be set to 4
#SBATCH --cpus-per-task=24             # Divide the number of cpus (96) by the number of GPUs (4)
#SBATCH --time=00:15:00
#SBATCH --partition=booster
#SBATCH --account=training2324
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --reservation=ai_on_sc_day2

export CUDA_VISIBLE_DEVICES=0,1,2,3    # Very important to make the GPUs visible
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

source $HOME/course/$USER/sc_venv_template/activate.sh
time srun python3 tb.py