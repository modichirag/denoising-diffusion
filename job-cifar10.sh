#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --constraint=a100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
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

dataset='cifar10'
channels=32
trainsteps=50_000
transport=1

time python -u deconvolving_interpolants.py \
                 --dataset $dataset --corruption $corruption \
                --corruption_level $clevel $nlevel $mlevel --train_steps $trainsteps \
                --channels $channels  --ode_steps 64 --alpha 0.9 --resamples 2  \
                --learning_rate 0.0005  --lr_scheduler --transport_steps $transport \
                --suffix "v3-32" --cond_y
#--cleansteps 128  --combinedsde  --gamma_scale 0.05
                #--load_model_path 'cifar10-random_motion-5.00-0.00-cds128-g0.01-combined-v3-32/model-best.pt'

#--suffix "v3-32"  --cleansteps 128   
                #--suffix "v3-32" --combinedsde --cleansteps 128 --save_transport --gamma_scale 0.005 --sampler "heun" 

#                --gamma_scale 0.05 --smodel --suffix "v3" --sampler "heun"

# time python -u deconvolving_interpolants.py \
#                  --dataset $dataset --corruption $corruption \
#                 --corruption_level $clevel $nlevel $mlevel --train_steps $trainsteps \
#                 --channels $channels  --ode_steps 64 --alpha 0.9 --resamples 2  \
#                 --gamma_scale 0.1 --diffusion_coeff 0.025 --smodel --transport_steps 1 \
#                 --suffix "v3" --learning_rate 0.0001 --cleansteps 10000 --save_every 1000
# #                --lr_scheduler --suffix "v3" --learning_rate 0.0005 \
