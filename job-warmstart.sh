#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --constraint=a100-80gb
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH -o logs/r%x.o%j

export OMP_NUM_THREADS=1

module load modules/2.3-20240529
source /mnt/home/cmodi/envs/torchlatest/bin/activate

corruption=$1
clevel=$2
nlevel=$3
mlevel=$4
echo $corruption
echo $clevel
echo $nlevel
datafolder='/cifar10-random_mask-0.50-0.00-1.00-v3/cleaned_best/'

channels=64
trainsteps=10_000
time python -u warmstart_deconvolving_interpolants.py  \
                --datafolder $datafolder --corruption $corruption \
                --corruption_level $clevel $nlevel $mlevel --train_steps $trainsteps \
                --channels $channels  --ode_steps 64 --alpha 0.9 --resamples 2  \
                --learning_rate 0.0005  --lr_scheduler --transport_steps 32 --cleansteps 2000 \
                --gamma_scale 0.05 --smodel --sampler "heun" 
