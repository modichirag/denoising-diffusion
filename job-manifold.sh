#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
##SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --constraint='a100'
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=14:00:00
#SBATCH -o logs/r%x.%j.out

export OMP_NUM_THREADS=1
source /mnt/home/${USER}/.zshrc
conda activate /mnt/home/jhan/miniforge3/envs/edm

python -u mlp_interpolants_trainer.py --corruption_level 2 0.01 --train_steps 200000 --learning_rate 1e-4 --suffix smalllr2