import torch
import sys, os
import json
import argparse
from torch.utils.data import DataLoader, Dataset
from ema_pytorch import EMA
import numpy as np

sys.path.append('./src/')
from networks import ConditionalDhariwalUNet
from custom_datasets import dataset_dict, ImagesOnly, cifar10_inverse_transforms
from interpolant_utils import DeconvolvingInterpolant, VelocityField
import forward_maps as fwd_maps
from utils import remove_orig_mod_prefix
from tqdm.auto import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="")
parser.add_argument("--model", type=str, default="best", help="which model to load")
parser.add_argument("--dataset", type=str, default='cifar10', help="dataset")
parser.add_argument("--corruption", type=str, help="corruption")
parser.add_argument("--corruption_levels", type=float, nargs='+', help="corruption level")
parser.add_argument("--channels", type=int, default=64, help="number of channels in model")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--prefix", type=str, default='', help="prefix for folder name")
parser.add_argument("--suffix", type=str, default='', help="suffix for folder name")
parser.add_argument("--subfolder", type=str, default='', help="subfolder for folder name")
parser.add_argument("--gated", action='store_true', help="gated convolution if provided, else not")
parser.add_argument("--ode_steps", type=int, default=64, help="number of steps for ODE sampling")
parser.add_argument("--multiview", action='store_true', help="change corruption every epoch if provided, else not")
parser.add_argument("--max_pos_embedding", type=int, default=2, help="number of resamplings")
args = parser.parse_args()
print(args)
if args.multiview:
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/multiview/'
else:
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/singleview/'

# Parse arguments
dataset, D, nc = dataset_dict[args.dataset]
dataset = dataset()
dl = DataLoader(ImagesOnly(dataset), batch_size = args.batch_size, \
                    shuffle = False, pin_memory = True, num_workers = 1)
gated = args.gated
if gated: 
    args.suffix = f"{args.suffix}-gated" if args.suffix else "gated"

# Parse corruption arguments
corruption = args.corruption
corruption_levels = args.corruption_levels
try:
    fwd_func = fwd_maps.corruption_dict[corruption](*corruption_levels)
except Exception as e:
    print("Exception in loading corruption function : ", e)
    sys.exit()
cname = "-".join([f"{i:0.2f}" for i in corruption_levels])
folder = f"{args.dataset}-{corruption}-{cname}"
if args.prefix != "": folder = f"{args.prefix}-{folder}"
if args.suffix != "": folder = f"{folder}-{args.suffix}"
if args.subfolder != "": folder = f"{folder}/{args.subfolder}/"

folder = f"{BASEPATH}/{folder}/"
results_folder = f"{folder}/cleaned_{args.model}/"
os.makedirs(results_folder, exist_ok=True)
print(f"Models will be loaded from folder: {folder}")
use_latents, latent_dim = fwd_maps.parse_latents(corruption, D)
if use_latents:
    print("Will use latents of dimension: ", latent_dim)


deconvolver = DeconvolvingInterpolant(fwd_func, use_latents=use_latents, n_steps=args.ode_steps).to(device)
b = ConditionalDhariwalUNet(D, nc, nc, latent_dim=latent_dim, model_channels=args.channels, gated=gated, \
                            max_pos_embedding=args.max_pos_embedding, zero_emb_channels_bwd=False).to(device)
ema_b = EMA(b)
data = torch.load(f'{folder}/model-{args.model}.pt', weights_only=True)
try:
    b.load_state_dict(data['model'])
    ema_b.load_state_dict(data['ema'])
except Exception as e :
    print("Saved compiled model. Trying to load without compilation")
    cleaned_ckpt = remove_orig_mod_prefix(data['model'])
    b.load_state_dict(cleaned_ckpt)
    cleaned_ckpt = remove_orig_mod_prefix(data['ema'])
    ema_b.load_state_dict(cleaned_ckpt)    
b = ema_b.ema_model.to(device)


@torch.inference_mode()
def get_cleaned_samples(image):
    corrupted, latents = deconvolver.push_fwd(image, return_latents=True)
    latents = latents if use_latents else None
    clean = deconvolver.transport(b, corrupted, latents)
    return clean


x = []
i = 0
for _, image in enumerate(dl):
    clean = get_cleaned_samples(image.to(device))
    x.append(clean.cpu())
    i += 1
    if (i % 10) == 0: 
        x = np.concatenate(x, axis=0)
        print(f"Processed {i} batches", x.shape)
        np.save(f"{results_folder}/cleaned_{i//10}.npy", x)
        x = []
x = np.concatenate(x, axis=0)
print(f"Processed {i} batches", x.shape)
np.save(f"{results_folder}/cleaned_{i//10+1}.npy", x)


