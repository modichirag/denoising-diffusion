import numpy as np
import sys, os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from transformers import get_cosine_schedule_with_warmup
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('./src/')
from utils import count_parameters, grab, is_compiled_model
from karras_unet_1d import KarrasUnet1D
from custom_datasets import  NumpyArrayDataset
from interpolant_utils import DeconvolvingInterpolant
from quasars import qso_model,  qso_dataloader, callback
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--corruption_levels", type=float, nargs='+', help="corruption level")
parser.add_argument("--channels", type=int, default=32, help="number of channels in model")
parser.add_argument("--train_steps", type=int, default=101, help="number of channels in model")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--prefix", type=str, default='', help="prefix for folder name")
parser.add_argument("--suffix", type=str, default='', help="suffix for folder name")
parser.add_argument("--lr_scheduler", action='store_true', help="use scheduler if provided, else not")
parser.add_argument("--dataset_seed", type=int, default=42, help="corrupt dataset seed")
parser.add_argument("--ode_steps", type=int, default=64, help="ode steps")
parser.add_argument("--alpha", type=float, default=0.9, help="probability of using new data")
parser.add_argument("--resamples", type=int, default=1, help="number of resamplings")
parser.add_argument("--singleview", action='store_true', help="keep corruption same if provided, else change")
args = parser.parse_args()
print(args)

if args.singleview: 
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/singleview/'
else:
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/multiview/'
print(BASEPATH)

# Parse arguments
normalize = False
D = 1024
downsample_factor = 4
model_channels = args.channels #192
train_num_steps = args.train_steps
save_and_sample_every = min(500, int(train_num_steps//50))
batch_size = args.batch_size
lr = args.learning_rate 
lr_scheduler = args.lr_scheduler

# Load and construct dataset
spectra_all = [np.load(f"/mnt/ceph/users/cmodi/ML_data/qsos/spectra-2.75-3.25/spectra{i}.npy")[...] \
               for i in range(40)]
spectra_all = np.concatenate(spectra_all, axis=0)
print("Loaded spectra of shape : ", spectra_all.shape)
# Subsize 
subs = spectra_all.shape[-1]//D
i0, i1 = (spectra_all.shape[-1]%D)//2, -(spectra_all.shape[-1]%D)//2
spectra = torch.from_numpy(spectra_all[:, 1,  i0:i1:subs]).to(device)
wavelength = torch.from_numpy(10**spectra_all[0, 0,  i0:i1:subs]).to(device)
print("Shape of spectra to use : ", spectra.shape)
mean_spectra = spectra.mean(dim=0)
std_spectra = spectra.std(dim=0)
if not normalize:
    mean_spectra = torch.zeros_like(mean_spectra)
    std_spectra = torch.ones_like(std_spectra)

norm_spectra = (spectra - mean_spectra)/(std_spectra)
dataset = NumpyArrayDataset(norm_spectra.cpu().numpy())
dataloader = qso_dataloader(wavelength, dataset, mean_spectra=mean_spectra, std_spectra=std_spectra, \
                            downsample=downsample_factor, batch_size=args.batch_size)

# folder name to save
idloglamb, min_snr, max_snr = args.corruption_levels
folder = f"qso_res-{idloglamb}_snr-{min_snr}-{max_snr}"
if args.prefix != "": folder = f"{args.prefix}-{folder}"
if args.suffix != "": folder = f"{folder}-{args.suffix}"
results_folder = f"{BASEPATH}/{folder}/"
os.makedirs(results_folder, exist_ok=True)
print(f"Results will be saved in folder: {results_folder}")


def train_step(b, denoiser, opt, sched, use_latents=False, surrogate=None):
    x = dataloader(args.batch_size).to(device)
    x1s, l = denoiser.push_fwd(x, return_latents=True)
    l = l if use_latents else None
    loss_val = denoiser.loss_fn(b, x1s, l)
    loss_val.backward()
    torch.nn.utils.clip_grad_norm_(b.parameters(), 1.0)
    opt.step()
    if sched is not None: sched.step()
    opt.zero_grad()    
    res = { 'loss': loss_val.detach().cpu(),}
    return res


fwd_func = qso_model(wavelength, mean_spectra, std_spectra, downsample_factor=downsample_factor, \
                     inv_delta_loglambda=idloglamb, min_snr=min_snr, max_snr=max_snr)
use_latents, latent_dim = False, None
denoiser = DeconvolvingInterpolant(fwd_func, use_latents, n_steps=args.ode_steps, 
                                   alpha=args.alpha, resamples=args.resamples).to(device)
#b = KarrasUnet1D(seq_len=D//downsample_factor, channels=downsample_factor, \
#                dim=16, dim_max=32, num_blocks_per_stage=1, num_downsamples=2).to(device)
b = KarrasUnet1D(seq_len=D//downsample_factor, channels=downsample_factor, \
                  dim=16, num_blocks_per_stage=2, num_downsamples=3, attn_res=(32)).to(device)
b = torch.compile(b)
print(f"Number of parameters : {count_parameters(b)[0]:0.3f} million")

opt = torch.optim.AdamW(b.parameters(), lr, weight_decay=0.01, amsgrad=True, fused=True)
sched = None if not lr_scheduler else \
    get_cosine_schedule_with_warmup(opt, train_num_steps//20, train_num_steps) 

losses = []
rmse = []
start = time()
_ = train_step(b, denoiser, opt, sched, use_latents=use_latents)
print("Time to compile : ", time() - start)
pbar = tqdm(range(train_num_steps))
for step in pbar:
    res = train_step(b, denoiser, opt, sched, use_latents=use_latents)
    loss = res['loss'].detach().numpy().mean()
    losses.append(loss)    
    # ema.update()

    if (step % save_and_sample_every) == 0:
        print(f"Saving model at step {step}")
        if ((step % 5000) == 0) & (step > 0):
            torch.save(b.state_dict(), os.path.join(results_folder, f"model-{step}.pt"))
            if is_compiled_model(b):
                torch.save(b._orig_mod.state_dict(), os.path.join(results_folder,"model_clean-{step}.pt"))
        np.save(os.path.join(results_folder, f"losses.npy"), np.array(losses))
        
        istep = step // save_and_sample_every
        callback(istep, dataloader=dataloader, denoiser=denoiser, b=b, \
                        device=device, results_folder=results_folder )

    pbar.set_description(f'Loss: {loss:.4f}')  


step = 'fin'
print(f"Saving model at step {step}")
torch.save(b.state_dict(), os.path.join(results_folder, f"model-{step}.pt"))
try:
    torch.save(b._orig_mod.state_dict(), os.path.join(results_folder, f"model_clean-{step}.pt")) 
except:
    print("Model not compiled. Cannot save clean model")
np.save(os.path.join(results_folder, f"losses.npy"), np.array(losses))
callback(step, dataloader=dataloader, denoiser=denoiser, b=b, \
                results_folder=results_folder, device=device )
