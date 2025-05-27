#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
##SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --constraint=a100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH -o logs/r%x.o%j

export OMP_NUM_THREADS=1

module load modules/2.3-20240529
source /mnt/home/cmodi/envs/torchlatest/bin/activate

nlevel=0.1
clevel=0.5
corruption="random_mask"
# corruption="gaussian_blur"
# clevel=5
# corruption="random_motion"

time TORCH_DISTRIBUTED_DEBUG=DETAIL  torchrun --standalone --nproc_per_node=1  deconvolving_distributed.py  \
        --dataset 'cifar10' --corruption $corruption --corruption_level $clevel $nlevel \
        --ode_steps 64 --alpha 0.9 --resamples 2 \
        --train_steps 5000 --channels 32 --batch_size 128 --learning_rate 0.001  \
        --suffix testw1 --lr_scheduler 