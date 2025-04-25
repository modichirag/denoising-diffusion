import numpy as np
import torch
import  matplotlib.pyplot as plt
import sys, os
sys.path.append('./src/')
from diffusion_model import Unet, GaussianDiffusion
from custom_datasets import cifar10_train, ImagesOnly
from torch.utils.data import DataLoader, Dataset
from trainer import Trainer
from utils import count_parameters

print("Setup model")
model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        flash_attn = True
    )

total, trainable = count_parameters(model)
print(f"Total params:     {total:,}")
print(f"Trainable params: {trainable:,}")

print("Setup diffusion model")
diffusion = GaussianDiffusion(
        model,
        image_size = 32,
        timesteps = 1000,
        sampling_timesteps = 20# number of steps
    )


print("Start training")
folder = '/mnt/ceph/users/cmodi/diffusion_guidance/cifar10/'
os.makedirs(folder, exist_ok=True)
cifar10_train_images = ImagesOnly(cifar10_train)
tmp = Trainer(diffusion, cifar10_train_images,
              train_batch_size=128,
              train_lr=5e-4,
              lr_scheduler=True,
              train_num_steps=10_001,
              calculate_fid=True,
              num_fid_samples=1_000,
              num_workers=4,
              amp=True,
              mixed_precision_type='fp16',
              save_best_and_latest_only=True,
              results_folder=folder)

losses = tmp.train()
np.save(f"{folder}/losses", losses)
print(losses[-1])
print()
with torch.cuda.amp.autocast(dtype=torch.float16):        # ← here’s the autocasting
    sampled_images = diffusion.sample(batch_size = 4)
#sampled_images = diffusion.sample(batch_size = 4)
#print(sampled_images.shape)

