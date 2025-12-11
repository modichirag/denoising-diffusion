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
from interpolant_utils import DeconvolvingInterpolant, DeconvolvingInterpolantCombined
import forward_maps as fwd_maps
from utils import remove_orig_mod_prefix, remove_all_prefix
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
parser.add_argument("--gamma_scale", type=float, default=0., help="noise added to interpolant")
parser.add_argument("--diffusion_coeff", type=float, default=0., help="diffusion coeff for sde")
parser.add_argument("--transport_steps", type=int, default=1, help="update transport map every n steps")
parser.add_argument("--smodel", action='store_true', help="use sde model")
parser.add_argument("--cleansteps", type=int, default=-1, help="update transport map every n steps")
parser.add_argument("--load_model_path", type=str, default='', help="load model from path")
parser.add_argument("--sampler", type=str, default='euler', help="load model from path")
parser.add_argument("--combinedsde", action='store_true', help="learn combined drift for sde model")
parser.add_argument("--randomize_t", action='store_true', help="randomize time stepping")

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
if args.cleansteps != -1: folder = f"{folder}-cds{args.cleansteps}"
if args.transport_steps != 1: folder = f"{folder}-tr{args.transport_steps}"
if args.smodel: folder = f"{folder}-sde"
if args.gamma_scale != 0: folder = f"{folder}-g{args.gamma_scale:0.2f}"
#if args.diffusion_coeff != 0: folder = f"{folder}-dc{args.diffusion_coeff:0.3f}"
if args.smodel: folder = f"{folder}-dc{args.diffusion_coeff:0.3f}"
if args.sampler != 'euler': folder = f"{folder}-{args.sampler}"
if args.randomize_t: folder = f"{folder}-randt"
if args.combinedsde: folder = f"{folder}-combined"
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

if args.combinedsde:
    deconvolver = DeconvolvingInterpolantCombined(fwd_func, use_latents=use_latents, \
                                                n_steps=args.ode_steps, gamma_scale=args.gamma_scale, sampler=args.sampler,
                                                  randomize_time=args.randomize_t).to(device)
else:
    deconvolver = DeconvolvingInterpolant(fwd_func, use_latents=use_latents, \
                                          n_steps=args.ode_steps, gamma_scale=args.gamma_scale, diffusion_coeff=args.diffusion_coeff,
                                          sampler=args.sampler, randomize_time=args.randomize_t).to(device)


# Load models
for emb  in [True, False]:
    try:
        b = ConditionalDhariwalUNet(D, nc, nc, latent_dim=latent_dim, model_channels=args.channels, gated=gated, \
                                    max_pos_embedding=args.max_pos_embedding, zero_emb_channels_bwd=emb).to(device)
        ema_b = EMA(b)
        data = torch.load(f'{folder}/model-{args.model}.pt', weights_only=True)
        cleaned_ckpt = remove_orig_mod_prefix(data['model'])
        b.load_state_dict(cleaned_ckpt)
        cleaned_ckpt = remove_orig_mod_prefix(data['ema'])
        ema_b.load_state_dict(cleaned_ckpt)    
        b = ema_b.ema_model

        s = None
        if 's_ema' in data:
            s = ConditionalDhariwalUNet(D, nc, nc, latent_dim=latent_dim, model_channels=channels, max_pos_embedding=args.max_pos_embedding, zero_emb_channels_bwd=emb).to(device)
            emas = EMA(s)
            emas.load_state_dict(cleaned_ckpt['s_ema'])
            s = emas.ema_model
            assert args.gamma_scale != 0.
            diffusion_coeff = 'gamma' if args.diffusion_coeff == 0.0 else  args.diffusion_coeff            
        continue
    
    except Exception as e:
        print(e)

if s is None:
    print('score network loaded')

# # setup the model
# b = ConditionalDhariwalUNet(D, nc, nc, latent_dim=latent_dim, model_channels=args.channels, gated=gated, \
#                             max_pos_embedding=args.max_pos_embedding, zero_emb_channels_bwd=True).to(device)
# ema_b = EMA(b)
# data = torch.load(f'{folder}/model-{args.model}.pt', weights_only=True)
# try:
#     b.load_state_dict(data['model'])
#     ema_b.load_state_dict(data['ema'])
# except Exception as e :
#     print("Saved compiled model. Trying to load without compilation")
#     cleaned_ckpt = remove_orig_mod_prefix(data['model'])
#     b.load_state_dict(cleaned_ckpt)
#     cleaned_ckpt = remove_orig_mod_prefix(data['ema'])
#     ema_b.load_state_dict(cleaned_ckpt)    
# b = ema_b.ema_model.to(device)

# if ('s_ema' in data.keys()) and args.smodel:
#     print("SDE training")
#     s_model =  ConditionalDhariwalUNet(D, nc, nc, latent_dim=latent_dim,
#                             model_channels=args.channels, gated=gated, \
#                             max_pos_embedding=args.max_pos_embedding).to(device)
#     ema = EMA(s_model)
#     ema.load_state_dict(remove_all_prefix(data['s_ema']))
#     s_model.load_state_dict(ema.ema_model.state_dict())

#     # set other params
#     if args.gamma_scale == 0. :
#         print("WARNING: SCORE NETWORK give with gamma=0. Setting gamma to 1.")
#         args.gamma_scale = 1.
#     if args.diffusion_coeff == 0. :
#         # print("WARNING: SCORE NETWORK give with diffusion coeff=0. Setting it to value of gamma*0.25 i.e. ". args.gamma_scale * 0.25)
#         # args.diffusion_coeff = args.gamma_scale * 0.25
#         print("WARNING: SCORE NETWORK give with diffusion coeff=0. Setting it to value gamma at all times")
#         args.diffusion_coeff = "gamma" 
# else:
#     s_model = None



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


