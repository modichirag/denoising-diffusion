import sys, os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader, Dataset
import numpy as np

sys.path.append('./src/')
from utils import grab, cycle, count_parameters, infinite_dataloader
from custom_datasets import dataset_dict, ImagesOnly, CorruptedDataset
from networks import ConditionalDhariwalUNet
from interpolant_utils import DeconvolvingInterpolant
import forward_maps as fwd_maps
from trainer_si import Trainer
import argparse

BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/singleview/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--dataset", type=str, help="dataset")
parser.add_argument("--corruption", type=str, help="corruption")
parser.add_argument("--corruption_levels", type=float, nargs='+', help="corruption level")
parser.add_argument("--channels", type=int, default=32, help="number of channels in model")
parser.add_argument("--train_steps", type=int, default=101, help="number of channels in model")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--prefix", type=str, default='', help="prefix for folder name")
parser.add_argument("--suffix", type=str, default='', help="suffix for folder name")
parser.add_argument("--gated", action='store_true', help="gated convolution if provided, else not")
parser.add_argument("--lr_scheduler", action='store_true', help="use scheduler if provided, else not")

# Parse arguments
args = parser.parse_args()

print(args)
dataset, D, nc = dataset_dict[args.dataset]
image_dataset = ImagesOnly(dataset)
model_channels = args.channels #192
train_num_steps = args.train_steps
save_and_sample_every = int(train_num_steps//50)
batch_size = args.batch_size
lr = args.learning_rate 
gated = args.gated
if gated: 
    args.suffix = f"{args.suffix}-gated" if args.suffix else "gated"
lr_scheduler = args.lr_scheduler

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
deconvolver = DeconvolvingInterpolant(fwd_func, use_latents=use_latents, n_steps=80).to(device)
corrupt_dataset = CorruptedDataset(image_dataset, deconvolver.push_fwd, base_seed=42)
# dl = infinite_dataloader(DataLoader(corrupt_dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 1))

trainer = Trainer(model=b, 
        deconvolver=deconvolver, 
        dataset = corrupt_dataset,
        train_batch_size = batch_size,
        gradient_accumulate_every = 1,
        train_lr = lr,
        lr_scheduler = lr_scheduler,
        train_num_steps = train_num_steps,
        save_and_sample_every= save_and_sample_every,
        results_folder=results_folder, 
        )

trainer.train()

# def train_step(xn, b, denoiser, opt, sched):
#     # ts  = torch.rand(xn.shape[0]).to(device)
#     loss_val = denoiser.loss_fn(b, xn)  
#     # perform backprop
#     loss_val.backward()
#     opt.step()
#     sched.step()
#     opt.zero_grad()    
#     res = {'loss': loss_val.detach().cpu()}
#     return res


# model =  DhariwalUNet(D, nc, nc, model_channels=model_channels).to(device)
# b = VelocityField(model)
# dl = cycle(DataLoader(image_dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 1))
# denoiser = DeconvolvingInterpolant(fwd_func, n_steps=80).to(device)

# opt = torch.optim.Adam([{'params': b.parameters(), 'lr': lr} ])
# sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.9)

# losses = []
# from tqdm import tqdm
# pbar = tqdm(range(train_num_steps))
# for i in pbar:    
#     x = next(dl).to(device)
#     xn = denoiser.push_fwd(x)
#     res = train_step(xn, b, denoiser, opt, sched)
#     loss = res['loss'].detach().numpy().mean()    
#     losses.append(loss)    
#     pbar.set_description(f'Loss: {loss:.4f}')      
#     if i % save_and_sample_every == 0:
#         torch.save(b.state_dict(), f"{results_folder}/model.pt")
#         np.save(f"{results_folder}/losses", losses)
#         image = next(dl).to(device)
#         corrupted = denoiser.push_fwd(image)
#         clean = denoiser.transport(b, corrupted)
#         save_fig(i, image, corrupted, clean, results_folder, corruption_level)
#         print(f"Saved model at step {i}")

# torch.save(b.state_dict(), f"{results_folder}/model.pt")
# np.save(f"{results_folder}/losses", losses)

# # Save fig
# image = next(dl).to(device)
# corrupted = denoiser.push_fwd(image)
# clean = denoiser.transport(b, corrupted)
# save_fig(42, image, corrupted, clean, results_folder, corruption_level)