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
from fid_evaluation import FIDEvaluation, calculate_frechet_distance
from utils import infinite_dataloader, grab, num_to_groups
from tqdm.auto import tqdm


BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="")
parser.add_argument("--dataset", type=str, help="dataset")
parser.add_argument("--corruption", type=str, help="corruption")
parser.add_argument("--corruption_levels", type=float, nargs='+', help="corruption level")
parser.add_argument("--channels", type=int, default=32, help="number of channels in model")
parser.add_argument("--n_samples", type=int, default=10_000, help="Samples to evalaute FID")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--prefix", type=str, default='', help="prefix for folder name")
parser.add_argument("--suffix", type=str, default='', help="suffix for folder name")
parser.add_argument("--gated", action='store_true', help="gated convolution if provided, else not")
parser.add_argument("--ode_steps", type=int, default=80, help="number of steps for ODE sampling")


# Parse arguments
args = parser.parse_args()
print(args)
dataset, D, nc = dataset_dict[args.dataset]
dl = infinite_dataloader(DataLoader(ImagesOnly(dataset), 
                                    batch_size = args.batch_size, \
                                    shuffle = True, pin_memory = True, num_workers = 1))
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
folder = f"{BASEPATH}/{folder}/"
results_folder = f"{folder}/results"
os.makedirs(results_folder, exist_ok=True)
print(f"Results will be saved in folder: {results_folder}")
n = int(args.n_samples/1e3)
save_name = f"{results_folder}/fid_{n}k_{args.ode_steps}steps.json"
print(f"Results will be saved in file: {save_name}")

# Latents
if 'mask' in corruption:
    use_latents = True
    print("Will be using latents: ", use_latents)
    latent_dim = [1, D, D]
else:
    use_latents = False
    latent_dim = None

# Load model
deconvolver = DeconvolvingInterpolant(fwd_func, use_latents=use_latents, n_steps=args.ode_steps).to(device)
try:
    b = ConditionalDhariwalUNet(D, nc, nc, latent_dim=latent_dim, model_channels=args.channels, gated=gated)
    try:
        b.load_state_dict(torch.load(f'{folder}/model.pt', weights_only=True))
        b.to(device)
    except:
        data = torch.load(f'{folder}/model-latest.pt', weights_only=True)
        b.load_state_dict(data['model'])
        b.to(device)
        ema_b = EMA(b)
        ema_b.load_state_dict(data['ema'])
        b = ema_b.ema_model
except:
    b = VelocityField(ConditionalDhariwalUNet(D, nc, nc, model_channels=args.channels, gated=gated))
    try:
        b.load_state_dict(torch.load(f'{folder}/model.pt', weights_only=True))
        b.to(device)
    except:
        data = torch.load(f'{folder}/model-latest.pt', weights_only=True)
        b.load_state_dict(data['model'])
        b.to(device)
        ema_b = EMA(b)
        ema_b.load_state_dict(data['ema'])
        b = ema_b.ema_model


fid_scorer = FIDEvaluation(
    batch_size=args.batch_size,
    dl=dl,
    channels=nc,
    accelerator=None, #args.accelerator,
    stats_dir=results_folder,
    device=device,
    num_fid_samples=args.n_samples,
    inception_block_idx=2048
)    
if not fid_scorer.dataset_stats_loaded:
    fid_scorer.load_or_precalc_dataset_stats(force_calc=True)

@torch.inference_mode()
def get_cleaned_samples():
    image = next(dl).to(device)
    corrupted, latents = deconvolver.push_fwd(image, return_latents=True)
    latents = latents if use_latents else None
    clean = deconvolver.transport(b, corrupted, latents)
    return clean


batches = num_to_groups(fid_scorer.n_samples, fid_scorer.batch_size)
stacked_fake_features = []
print(f"Stacking Inception features for {fid_scorer.n_samples} generated samples.")

for batch in tqdm(batches):
    fake_samples = get_cleaned_samples()    
    fake_features = fid_scorer.calculate_inception_features(fake_samples)
    stacked_fake_features.append(fake_features)
stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
m1 = np.mean(stacked_fake_features, axis=0)
s1 = np.cov(stacked_fake_features, rowvar=False)
score = calculate_frechet_distance(m1, s1, fid_scorer.m2, fid_scorer.s2)
print(f"FID score of loaded best model : {score}")

to_save = {'FID_best': score}
with open(save_name, 'w') as file:
        json.dump(to_save, file, indent=4)
