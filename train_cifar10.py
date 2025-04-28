import numpy as np
import torch
import  matplotlib.pyplot as plt
import sys, os
sys.path.append('./src/')
from networks import EDMPrecond
from loss_functions import EDMLoss, VELoss, VPLoss
from custom_datasets import cifar10_train, ImagesOnly
from torch.utils.data import DataLoader, Dataset
from trainer import Trainer
from utils import count_parameters


D = 32
nc = 3
folder = '/mnt/ceph/users/cmodi/diffusion_guidance/cifar10-w2/'
os.makedirs(folder, exist_ok=True)

print("Setup model")
model = EDMPrecond(D, nc)
loss_fn = VELoss()
total, trainable = count_parameters(model)
print(f"Total params:     {total:,}")
print(f"Trainable params: {trainable:,}")


print("Start training")
cifar10_train_images = ImagesOnly(cifar10_train)
tmp = Trainer(model, loss_fn, cifar10_train_images,
              train_batch_size = 128,
              train_lr = 5e-4,
              lr_scheduler = True,
              train_num_steps = 5_001,
              save_and_sample_every = 5_00,
              calculate_fid = True,
              num_fid_samples = 2_000,
              num_workers = 1,
              amp = True,
              mixed_precision_type = 'bf16',
              save_best_and_latest_only = True,
              results_folder = folder)

losses = tmp.train()
np.save(f"{folder}/losses", losses)
print(losses[-1])
print()
# with torch.cuda.amp.autocast(dtype=torch.float16):        # ← here’s the autocasting
#     sampled_images = diffusion.sample(batch_size = 4)
#sampled_images = diffusion.sample(batch_size = 4)
#print(sampled_images.shape)

