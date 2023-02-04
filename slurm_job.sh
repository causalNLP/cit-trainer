#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=102400
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:10g
#SBATCH --time=24:00:00

#SBATCH -J myFirstGPUJob
#SBATCH -o job.%j.out

source ~/.bashrc
conda activate ci_train
python experiment.py --dataset_size 128000 --lambda_ci 0.1