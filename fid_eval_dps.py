import torch
import sys, os
import json
import argparse
from torch.utils.data import DataLoader, Dataset
from ema_pytorch import EMA
import numpy as np

sys.path.append('./src/')
from networks import EDMPrecond
from custom_datasets import dataset_dict, ImagesOnly, cifar10_inverse_transforms
from interpolant_utils import DeconvolvingInterpolant
import forward_maps as fwd_maps
from fid_evaluation import FIDEvaluation, calculate_frechet_distance
from utils import infinite_dataloader,  num_to_groups, remove_orig_mod_prefix
from tqdm.auto import tqdm
from dps import edm_sampler_dps

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="")
parser.add_argument("--model", type=str, default="best", help="which model to load")
parser.add_argument("--modelfolder", type=str, default="cifar10-96-long", help="which model to load")
parser.add_argument("--dataset", type=str, help="dataset")
parser.add_argument("--corruption", type=str, help="corruption")
parser.add_argument("--corruption_levels", type=float, nargs='+', help="corruption level")
parser.add_argument("--channels", type=int, default=96, help="number of channels in model")
parser.add_argument("--n_samples", type=int, default=50_000, help="Samples to evalaute FID")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--prefix", type=str, default='', help="prefix for folder name")
parser.add_argument("--suffix", type=str, default='', help="suffix for folder name")
parser.add_argument("--subfolder", type=str, default='', help="subfolder for folder name")
parser.add_argument("--gated", action='store_true', help="gated convolution if provided, else not")
parser.add_argument("--multiview", action='store_true', help="change corruption every epoch if provided, else not")
parser.add_argument("--max_pos_embedding", type=int, default=2, help="number of resamplings")
parser.add_argument("--num_steps", type=int, default=256, help="number of diffusion steps")
parser.add_argument("--Schurn", type=int, default=30, help="Schurn")
parser.add_argument("--conditioning_scale", type=float, default=1.0, help="conditioning scale")
args = parser.parse_args()
print(args)
if args.multiview:
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/multiview/'
else:
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/singleview/'

# Parse arguments
dataset, D, nc = dataset_dict[args.dataset]
dataset = dataset()
model_channels = args.channels #192
print(D, nc)
dl = infinite_dataloader(DataLoader(ImagesOnly(dataset), 
                                    batch_size = args.batch_size, \
                                    shuffle = True, pin_memory = True, num_workers = 1))
gated = args.gated
if gated: 
    args.suffix = f"{args.suffix}-gated" if args.suffix else "gated"
print(BASEPATH)

# Parse corruption arguments
corruption = args.corruption
corruption_levels = args.corruption_levels
corruption_levels_zero = [args.corruption_levels[0]] + [0.]*(len(corruption_levels)-1)
print(corruption_levels, corruption_levels_zero)
try:
    fwd_func = fwd_maps.corruption_dict[corruption](*corruption_levels)
    fwd_func_zero = fwd_maps.corruption_dict[corruption](*corruption_levels_zero)
except Exception as e:
    print("Exception in loading corruption function : ", e)
    sys.exit()

cname = "-".join([f"{i:0.2f}" for i in corruption_levels])
folder = f"{args.dataset}-{corruption}-{cname}"
if args.prefix != "": folder = f"{args.prefix}-{folder}"
if args.suffix != "": folder = f"{folder}-{args.suffix}"
if args.subfolder != "": folder = f"{folder}/{args.subfolder}/"

folder = f"{BASEPATH}/{folder}/"
results_folder = f"{folder}/results-dps/"
os.makedirs(results_folder, exist_ok=True)
print(f"Models will be loaded from folder: {folder}")
use_latents, latent_dim = fwd_maps.parse_latents(corruption, D)
if use_latents:
    print("Will use latents of dimension: ", latent_dim)
n = int(args.n_samples/1e3)
save_name = f"{results_folder}/fid_{n}k_{args.num_steps}steps_Sch{args.Schurn}_cscale{args.conditioning_scale:0.1f}.json"
print(f"Results will be saved in file: {save_name}")


model = EDMPrecond(D, nc, model_channels=model_channels).to(device)
ema = EMA(model).to(device)
data = torch.load(f'/mnt/ceph/users/cmodi/diffusion_guidance/{args.modelfolder}/model-{args.model}.pt',\
                             map_location=device, weights_only=True)
ema.load_state_dict(data['ema'])
ema_model = ema.ema_model
print("Model loaded")
del ema, model, data

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


#@torch.inference_mode()
def get_cleaned_samples():
    image = next(dl).to(device)
    corrupted, latents = fwd_func(image, return_latents=True)
    latents = latents if use_latents else None
    z = torch.randn_like(corrupted)
    clean = edm_sampler_dps(net=ema_model, latents=z, fwd_func=fwd_func_zero, data=corrupted, data_latents=latents, 
                            num_steps=args.num_steps, S_churn=args.Schurn, conditioning_scale=args.conditioning_scale)
    del image, corrupted, z
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
