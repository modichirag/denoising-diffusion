import numpy as np
import sys, os
import torch
import torch.distributed as dist
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
import json

sys.path.append('./src/')
from utils import count_parameters, make_serializable
from karras_unet_1d import KarrasUnet1D, PixelUnShuffle1D
from custom_datasets import  NumpyArrayDataset, CorruptedDataset
from interpolant_utils import DeconvolvingInterpolant, DeconvolvingInterpolantCombined
from trainer_si import Trainer, get_worker_info
from quasars import qso_model,  qso_dataloader, qso_callback
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--corruption_levels", type=float, nargs='+', help="corruption level")
parser.add_argument("--train_steps", type=int, default=101, help="number of channels in model")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--mini_batch_size", type=int, default=128, help="batch size per iteration")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--prefix", type=str, default='', help="prefix for folder name")
parser.add_argument("--suffix", type=str, default='', help="suffix for folder name")
parser.add_argument("--lr_scheduler", action='store_true', help="use scheduler if provided, else not")
parser.add_argument("--dataset_seed", type=int, default=42, help="corrupt dataset seed")
parser.add_argument("--ode_steps", type=int, default=64, help="ode steps")
parser.add_argument("--alpha", type=float, default=0.9, help="probability of using new data")
parser.add_argument("--resamples", type=int, default=1, help="number of resamplings")
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
print(BASEPATH)

# Initialize DDP
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
world_size, rank, local_rank, device = get_worker_info()
torch.cuda.set_device(device)
print(f"DEVICE (Rank {local_rank}): {device}")


# Parse arguments
normalize = False
D = 1024
downsample_factor = 4
train_num_steps = args.train_steps
save_and_sample_every = min(500, int(train_num_steps//50))
batch_size = args.batch_size
lr = args.learning_rate 
lr_scheduler = args.lr_scheduler
train_batch_size = int(batch_size//world_size)
gradient_accumulate_every = max(1, int(train_batch_size // args.mini_batch_size))


# Load and construct dataset
spectra_all = [np.load(f"/mnt/ceph/users/cmodi/ML_data/qsos/spectra-2.75-3.25/spectra{i}.npy")[...] \
               for i in range(40)] 
spectra_all = np.concatenate(spectra_all, axis=0)
print("Loaded spectra of shape : ", spectra_all.shape)
# Subsize 
subs = spectra_all.shape[-1]//D
i0, i1 = (spectra_all.shape[-1]%D)//2, -(spectra_all.shape[-1]%D)//2
spectra = spectra_all[:, 1,  i0:i1:subs]
wavelength = torch.from_numpy(10**spectra_all[0, 0,  i0:i1:subs])#.to(device)
print("Shape of spectra to use : ", spectra.shape)
transform = lambda x: PixelUnShuffle1D(downsample_factor)(x.unsqueeze(0))
dataset = NumpyArrayDataset(spectra, transform=transform)
qdataloader = qso_dataloader(wavelength, dataset, \
                            downsample=downsample_factor, batch_size=args.batch_size)


# Parse corruption arguments
idloglamb, min_snr, max_snr = args.corruption_levels
fwd_func = qso_model(wavelength, downsample_factor=downsample_factor, \
                     inv_delta_loglambda=idloglamb, min_snr=min_snr, max_snr=max_snr)
use_latents, latent_dim = False, None


# Folder name 
folder = f"qso_res-{idloglamb}_snr-{min_snr}-{max_snr}"
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
results_folder = f"{BASEPATH}/{folder}/"
os.makedirs(results_folder, exist_ok=True)
args_dict = make_serializable(vars(args) if isinstance(args, argparse.Namespace) else args)
print(f"Results will be saved in folder: {results_folder}")
if local_rank == 0:
    with open(f"{results_folder}/args.json", "w") as f:
        json.dump(args_dict, f, indent=4)


# Initialize model and train
# b = KarrasUnet1D(seq_len=D//downsample_factor, channels=downsample_factor, \
#                dim=16, dim_max=32, num_blocks_per_stage=1, num_downsamples=2).to(device)
b = KarrasUnet1D(seq_len=D//downsample_factor, channels=downsample_factor, \
                  dim=16, num_blocks_per_stage=2, num_downsamples=3, attn_res=(32)).to(device)
if args.smodel:
    print("SDE training")
    s_model =  ConditionalDhariwalUNet(D, nc, nc, latent_dim=latent_dim,
                            model_channels=model_channels, gated=gated, \
                            max_pos_embedding=args.max_pos_embedding).to(device)
    if args.gamma_scale == 0. :
        print("WARNING: SCORE NETWORK give with gamma=0. Setting gamma to 1.")
        args.gamma_scale = 1.
    if args.diffusion_coeff == 0. :
        print("WARNING: SCORE NETWORK give with diffusion coeff=0. Setting it to value gamma at all times")
        args.diffusion_coeff = "gamma"
else:
    s_model = None

print(f"Number of parameters : {count_parameters(b)[0]:0.3f} million")
if args.combinedsde:
    deconvolver = DeconvolvingInterpolantCombined(fwd_func, use_latents=use_latents, \
                                      alpha=args.alpha, resamples=args.resamples, n_steps=args.ode_steps, \
                                                  gamma_scale=args.gamma_scale, sampler=args.sampler,
                                                  randomize_time=args.randomize_t).to(device)
else:
    deconvolver = DeconvolvingInterpolant(fwd_func, use_latents=use_latents, \
                                      alpha=args.alpha, resamples=args.resamples, n_steps=args.ode_steps, \
                                      gamma_scale=args.gamma_scale, diffusion_coeff=args.diffusion_coeff,
                                          sampler=args.sampler, randomize_time=args.randomize_t).to(device)

corrupt_dataset = CorruptedDataset(dataset, fwd_func, \
                                   tied_rng=not(args.multiview), base_seed=args.dataset_seed)
dataset_sampler = DistributedSampler(corrupt_dataset, num_replicas=world_size, \
                                     shuffle=True, rank=local_rank)

print("Launch Train")
trainer = Trainer(model=b, 
                  ddp = ddp,
                  compile_model=True,
                  deconvolver=deconvolver, 
                  dataset = corrupt_dataset,
                  dataset_sampler=dataset_sampler,
                  train_batch_size = train_batch_size,
                  gradient_accumulate_every = gradient_accumulate_every,
                  train_lr = lr,
                  lr_scheduler = lr_scheduler,
                  train_num_steps = train_num_steps,
                  save_and_sample_every= save_and_sample_every,
                  results_folder=results_folder, 
                  warmup_fraction=0.05,
                  update_transport_every=args.transport_steps,
                  callback_fn = qso_callback,
                  callback_kwargs = {"qdataloader": qdataloader},
                  s_model=s_model,
                  clean_data_steps=args.cleansteps
                  # mixed_precision_type = 'fp32',
            )

trainer.train()

dist.destroy_process_group()

# opt = torch.optim.AdamW(b.parameters(), lr, weight_decay=0.01, amsgrad=True, fused=True)
# sched = None if not lr_scheduler else \
#     get_cosine_schedule_with_warmup(opt, train_num_steps//20, train_num_steps) 

# def train_step(b, denoiser, opt, sched, use_latents=False, surrogate=None):
#     x = dataloader(args.batch_size).to(device)
#     x1s, l = denoiser.push_fwd(x, return_latents=True)
#     l = l if use_latents else None
#     loss_val = denoiser.loss_fn(b, x1s, l)
#     loss_val.backward()
#     torch.nn.utils.clip_grad_norm_(b.parameters(), 1.0)
#     opt.step()
#     if sched is not None: sched.step()
#     opt.zero_grad()    
#     res = { 'loss': loss_val.detach().cpu(),}
#     return res


# losses = []
# rmse = []
# start = time()
# _ = train_step(b, denoiser, opt, sched, use_latents=use_latents)
# print("Time to compile : ", time() - start)
# pbar = tqdm(range(train_num_steps))
# for step in pbar:
#     res = train_step(b, denoiser, opt, sched, use_latents=use_latents)
#     loss = res['loss'].detach().numpy().mean()
#     losses.append(loss)    
#     # ema.update()

#     if (step % save_and_sample_every) == 0:
#         print(f"Saving model at step {step}")
#         if ((step % 5000) == 0) & (step > 0):
#             torch.save(b.state_dict(), os.path.join(results_folder, f"model-{step}.pt"))
#             if is_compiled_model(b):
#                 torch.save(b._orig_mod.state_dict(), os.path.join(results_folder,"model_clean-{step}.pt"))
#         np.save(os.path.join(results_folder, f"losses.npy"), np.array(losses))
        
#         istep = step // save_and_sample_every
#         callback(istep, dataloader=dataloader, denoiser=denoiser, b=b, \
#                         device=device, results_folder=results_folder )

#     pbar.set_description(f'Loss: {loss:.4f}')  


# step = 'fin'
# print(f"Saving model at step {step}")
# torch.save(b.state_dict(), os.path.join(results_folder, f"model-{step}.pt"))
# try:
#     torch.save(b._orig_mod.state_dict(), os.path.join(results_folder, f"model_clean-{step}.pt")) 
# except:
#     print("Model not compiled. Cannot save clean model")
# np.save(os.path.join(results_folder, f"losses.npy"), np.array(losses))
# callback(step, dataloader=dataloader, denoiser=denoiser, b=b, \
#                 results_folder=results_folder, device=device )
