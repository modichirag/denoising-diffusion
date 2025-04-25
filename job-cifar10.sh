#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --constraint=h100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:20:00
#SBATCH -o logs/r%x.o%j

export OMP_NUM_THREADS=1

module load modules/2.3-20240529
source /mnt/home/cmodi/envs/torchlatest/bin/activate

#time torchrun run.py --config configs/finewebmini.yaml
time python -u train_cifar10.py
