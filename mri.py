import sys, os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

sys.path.append('./src/')
from utils import count_parameters
from custom_datasets import dataset_dict, ImagesOnly, CorruptedDataset
from custom_datasets import CombinedNumpyDataset, CombinedLazyNumpyDataset
from networks import ConditionalDhariwalUNet
from interpolant_utils import DeconvolvingInterpolant
import forward_maps as fwd_maps
from forward_maps import mri_subsampling, mri_subsampling2, mri_subsampling3
from trainer_si import Trainer
import argparse


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--corruption", type=str, help="corruption")
parser.add_argument("--mode", type=str, default='same_rate', help="corruption")
parser.add_argument("--corruption_levels", type=float, nargs='+', help="corruption level")
parser.add_argument("--channels", type=int, default=32, help="number of channels in model")
parser.add_argument("--train_steps", type=int, default=101, help="number of channels in model")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--prefix", type=str, default='', help="prefix for folder name")
parser.add_argument("--suffix", type=str, default='', help="suffix for folder name")
parser.add_argument("--gated", action='store_true', help="gated convolution if provided, else not")
parser.add_argument("--lr_scheduler", action='store_true', help="use scheduler if provided, else not")
parser.add_argument("--dataset_seed", type=int, default=42, help="corrupt dataset seed")
parser.add_argument("--ode_steps", type=int, default=64, help="ode steps")
parser.add_argument("--alpha", type=float, default=0.9, help="probability of using new data")
parser.add_argument("--resamples", type=int, default=1, help="number of resamplings")
parser.add_argument("--multiview", action='store_true', help="change corruption every epoch if provided, else not")

args = parser.parse_args()
print(args)

if args.multiview: 
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/multiview/'
else:
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/singleview/'

D0 = 160
s = 4
D = int(D0/s)
nc = int(s**2)
if args.corruption == "mri2": nc *= 2
train_batch_size = 64

# Parse arguments
model_channels = args.channels #192
train_num_steps = args.train_steps
save_and_sample_every = min(200, int(train_num_steps//50))
batch_size = args.batch_size
lr = args.learning_rate 
gated = args.gated
if gated: 
    args.suffix = f"{args.suffix}-gated" if args.suffix else "gated"
lr_scheduler = args.lr_scheduler

# Parse corruption arguments
if D0 == 160:
    data_folder = "/mnt/ceph/users/cmodi/ML_data/fastMRI/knee-singlecoil-train-fourier-sub//"
else:
    data_folder = "/mnt/ceph/users/cmodi/ML_data/fastMRI/knee-singlecoil-train-fourier/"
dataset = CombinedNumpyDataset(data_folder)
corruption = args.corruption
corruption_levels = args.corruption_levels
if args.corruption == 'mri1':
    fwd_func = mri_subsampling1(**corruption_levels, mode=mode)
if args.corruption == 'mri2':
    fwd_func = mri_subsampling2(**corruption_levels, mode=mode)
if args.corruption == 'mri3':
    fwd_func = mri_subsampling3(**corruption_levels, mode=mode)
# try:
#     fwd_func = fwd_maps.corruption_dict[corruption](*corruption_levels)
# except Exception as e:
#     print("Exception in loading corruption function : ", e)
#     sys.exit()
cname = "-".join([f"{i:0.2f}" for i in corruption_levels])
folder = f"mri-{corruption}-{cname}"
if args.prefix != "": folder = f"{args.prefix}-{folder}"
if args.suffix != "": folder = f"{folder}-{args.suffix}"
results_folder = f"{BASEPATH}/{folder}/"
os.makedirs(results_folder, exist_ok=True)
print(f"Results will be saved in folder: {results_folder}")
use_latents, latent_dim = fwd_maps.parse_latents(corruption, D)
if use_latents:
    print("Will use latents of dimension: ", latent_dim)

# Initialize model and train
b =  ConditionalDhariwalUNet(D, nc, nc, latent_dim=latent_dim,
                        model_channels=model_channels, gated=gated).to(device)
print("Parameter count : ", count_parameters(b))
deconvolver = DeconvolvingInterpolant(fwd_func, use_latents=use_latents, \
                                      alpha=args.alpha, resamples=args.resamples, n_steps=args.ode_steps).to(device)
corrupt_dataset = CorruptedDataset(dataset, deconvolver.push_fwd, \
                                   tied_rng=not(args.multiview), base_seed=args.dataset_seed)

trainer = Trainer(model=b, 
                  deconvolver=deconvolver, 
                  dataset = corrupt_dataset,
                  train_batch_size = train_batch_size,
                  gradient_accumulate_every = args.batch_size//train_batch_size,
                  train_lr = lr,
                  lr_scheduler = lr_scheduler,
                  train_num_steps = train_num_steps,
                  save_and_sample_every= save_and_sample_every,
                  results_folder=results_folder, 
                  warmup_fraction=0.05
        )

trainer.train()

