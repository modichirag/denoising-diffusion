#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
##SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --constraint=a100
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH -o logs/r%x.o%j

export OMP_NUM_THREADS=1

module load modules/2.3-20240529
source /mnt/home/cmodi/envs/torchlatest/bin/activate

#time python -u train_mnist.py
time python -u fid_eval.py /mnt/ceph/users/cmodi/diffusion_guidance/mnist/
