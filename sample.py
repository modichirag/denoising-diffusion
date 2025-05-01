import numpy as np
import torch
import  matplotlib.pyplot as plt
import sys, os
from tqdm import tqdm
from ema_pytorch import EMA

sys.path.append('./src/')
from networks import EDMPrecond
from custom_datasets import dataset_dict
from generate import edm_sampler, edm_fine
from utils import num_to_groups
import argparse

BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/'

parser = argparse.ArgumentParser(description="")
parser.add_argument("--folder", type=str, help="Path")
parser.add_argument("--dataset", type=str, help="dataset")
parser.add_argument("--sigma_min", type=float, default=0.0, help="min VE sigma")
parser.add_argument("--channels", type=int, default=32, help="number of channels in model")
parser.add_argument("--train_steps", type=int, default=20_001, help="number of channels in model")
parser.add_argument("--num_samples", type=int, default=5_001, help="number of samples to generate")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--seed", type=int, default=42, help="sampling seed")
parser.add_argument("--sde_sampling", action='store_true', help="sde sampling if provided, else determinisitic")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

# Parse arguments
args = parser.parse_args()
print(args)
folder = f"{BASEPATH}/{args.folder}/"
dataset, D, nc = dataset_dict[args.dataset]
model_channels = args.channels #192
sigma_min = args.sigma_min
num_samples = args.num_samples
batch_size = args.batch_size
torch.manual_seed(args.seed)
if device == 'cuda':
    torch.cuda.manual_seed_all(args.seed)
extrap_to_zero_time = True if sigma_min == 0. else False

# Create folder to save samples
save_folder = f"{BASEPATH}/{args.folder}/samples/"
os.makedirs(save_folder, exist_ok=True)
print(f"Saving samples to {save_folder}")

# Model
print("Setup model")
model = EDMPrecond(D, nc, model_channels=model_channels)
ema = EMA(model).to(device)
del model
data = torch.load(f'{folder}/model-best.pt', map_location=device, weights_only=True)
ema.load_state_dict(data['ema'])
ema_model = ema.ema_model
ema_model.eval()

# Sample
print("Start sampling")
if args.sde_sampling:
    save_name = f"samples-{sigma_min:0.3f}-sde.npy"
    print("Enabled SDE sampling. This will be slow.") 
else:
    save_name = f"samples-{sigma_min:0.3f}.npy"
batches = num_to_groups(num_samples, batch_size)
all_samples = []
with torch.no_grad():
    for batch in tqdm(batches):
        latents = torch.randn(size=(batch, nc, D, D), device=device)
        if args.sde_sampling:
            if len(all_samples): #save intermediate samples
                np.save(os.path.join(save_folder, save_name), \
                    torch.cat(all_samples, dim=0).cpu().numpy()) 
            samples = edm_sampler(ema_model, latents, **edm_fine, sigma_min=sigma_min, extrap_to_zero_time=extrap_to_zero_time)
        else:
            samples = edm_sampler(ema_model, latents, sigma_min=sigma_min, extrap_to_zero_time=extrap_to_zero_time)
        all_samples.append(samples)
        
# Save all samples
all_samples = torch.cat(all_samples, dim=0).cpu().numpy()
np.save(os.path.join(save_folder, save_name), all_samples)