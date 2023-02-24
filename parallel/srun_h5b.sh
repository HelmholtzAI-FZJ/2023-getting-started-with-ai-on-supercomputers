#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --output=/p/home/jusers/benassou1/juwels/benassou1/2023-getting-started-with-ai-on-supercomputers/parallel/output/%j.out
#SBATCH --error=/p/home/jusers/benassou1/juwels/benassou1/2023-getting-started-with-ai-on-supercomputers/parallel/output/%j.err
#SBATCH --time=23:59:59
#SBATCH --partition=booster
#SBATCH --gres=gpu:1
#SBATCH --account=atmlaml

source sc_venv_template/activate.sh

srun python -u to_h5.py