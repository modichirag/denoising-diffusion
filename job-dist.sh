#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=4
##SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --constraint=a100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
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
echo $mlevel

dataset='celebA'
channels=96 
trainsteps=50_000

time TORCH_DISTRIBUTED_DEBUG=DETAIL  \
torchrun --standalone --nproc_per_node=4  deconvolving_distributed.py  \
     --dataset $dataset --train_steps $trainsteps --channels $channels \
     --corruption $corruption --corruption_level $clevel $nlevel $mlevel \
     --ode_steps 64 --alpha 0.9 --resamples 2 \
     --batch_size 128 --learning_rate 0.0005  --lr_scheduler --suffix "v2"
