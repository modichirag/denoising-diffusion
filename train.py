import numpy as np
import torch
import  matplotlib.pyplot as plt
import sys, os
sys.path.append('./src/')
from networks import EDMPrecond
from loss_functions import EDMLoss, VELoss, VPLoss
from custom_datasets import dataset_dict, ImagesOnly
from torch.utils.data import DataLoader, Dataset
from trainer import Trainer
from utils import count_parameters
import argparse

BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/'

parser = argparse.ArgumentParser(description="")
parser.add_argument("--folder", type=str, help="Path")
parser.add_argument("--dataset", type=str, help="dataset")
parser.add_argument("--sigma_min", type=float, default=0.02, help="min VE sigma")
parser.add_argument("--channels", type=int, default=64, help="number of channels in model")
parser.add_argument("--train_steps", type=int, default=20_001, help="number of channels in model")

# Parse arguments
args = parser.parse_args()
print(args)
results_folder = f"{BASEPATH}/{args.folder}/"
os.makedirs(results_folder, exist_ok=True)
dataset, D, nc = dataset_dict[args.dataset]
image_dataset = ImagesOnly(dataset)
model_channels = args.channels #192
train_num_steps = args.train_steps
save_and_sample_every = int(train_num_steps//10)
sigma_min = args.sigma_min

# Model
print("Setup model")
model = EDMPrecond(D, nc, model_channels=model_channels)
loss_fn = VELoss(sigma_min=sigma_min)
total, trainable = count_parameters(model)
print(f"Total params:     {total:,}")
print(f"Trainable params: {trainable:,}")

# Train
print("Start training")
tmp = Trainer(model, loss_fn, image_dataset,
              train_batch_size = 256,
              train_lr = 3e-4,
              lr_scheduler = False,
              train_num_steps = train_num_steps,
              save_and_sample_every = save_and_sample_every,
              calculate_fid = True,
              num_fid_samples = 2_000,
              num_workers = 1,
              amp = True,
              mixed_precision_type = 'bf16',
              save_best_and_latest_only = True,
              results_folder = results_folder)

losses, fids = tmp.train()
np.save(f"{results_folder}/losses", losses)
np.save(f"{results_folder}/fids", fids)
print(losses[-1])
print()

