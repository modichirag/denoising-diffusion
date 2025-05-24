#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=2
##SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --constraint=h100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH -o logs/r%x.o%j

export OMP_NUM_THREADS=1

module load modules/2.3-20240529
source /mnt/home/cmodi/envs/torchlatest/bin/activate

clevel=$1
nlevel=$2
corruption="random_mask"
# corruption="block_mask"
# corruption="gaussian_blur"

torchrun --standalone --nproc_per_node=2  deconvolving_distributed.py  \
        --dataset cifar10 --corruption $corruption --corruption_level $clevel $nlevel \
        --train_steps 5_000 --channels 64 --batch_size 256 --learning_rate 0.001  \
        --suffix condtest --lr_scheduler 

