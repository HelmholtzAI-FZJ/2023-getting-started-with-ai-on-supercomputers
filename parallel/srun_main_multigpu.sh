#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --output=/p/home/jusers/benassou1/juwels/benassou1/2023-getting-started-with-ai-on-supercomputers/parallel/output/%j.out
#SBATCH --error=/p/home/jusers/benassou1/juwels/benassou1/2023-getting-started-with-ai-on-supercomputers/parallel/output/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:4
#SBATCH --account=atmlaml

source sc_venv_template/activate.sh

srun python -u main_multigpu.py --data_dir "tiny-imagenet-200/" --log "/p/home/jusers/benassou1/juwels/benassou1/2023-getting-started-with-ai-on-supercomputers/parallel/logs/" 