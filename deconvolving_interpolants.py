import sys, os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader, Dataset
import numpy as np

sys.path.append('./src/')
from utils import grab, cycle
from custom_datasets import dataset_dict, ImagesOnly
from networks import DhariwalUNet
from interpolant_utils import VelocityField, DeconvolvingInterpolant, save_fig
import forward_maps as fwd_maps
from trainer_si import Trainer
import argparse

BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/'

parser = argparse.ArgumentParser(description="")
parser.add_argument("--dataset", type=str, help="dataset")
parser.add_argument("--corruption", type=str, default='gaussian_noise', help="corruption")
parser.add_argument("--corruption_level", type=float, default=0.1, help="corruption level")
parser.add_argument("--channels", type=int, default=32, help="number of channels in model")
parser.add_argument("--train_steps", type=int, default=101, help="number of channels in model")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--prefix", type=str, default='', help="prefix for folder name")
parser.add_argument("--suffix", type=str, default='', help="suffix for folder name")

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
lr_scheduler = True


# Parse corruption arguments
corruption = args.corruption
corruption_level = args.corruption_level
try:
    fwd_func = fwd_maps.corruption_dict[corruption](corruption_level)
except Exception as e:
    print(e)
    sys.exit()
folder = f"{args.dataset}-{corruption}-{corruption_level}"
if args.prefix != "": folder = f"{args.prefix}-{folder}"
if args.suffix != "": folder = f"{folder}-{args.suffix}"
results_folder = f"{BASEPATH}/{folder}/"
os.makedirs(results_folder, exist_ok=True)
print(f"Results will be saved in folder: {results_folder}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)


model =  DhariwalUNet(D, nc, nc, model_channels=model_channels).to(device)
b = VelocityField(model)
dl = cycle(DataLoader(image_dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 1))
deconvolver = DeconvolvingInterpolant(fwd_func, n_steps=80).to(device)

trainer = Trainer(model=b, 
        deconvolver=deconvolver, 
        dataset = image_dataset,
        train_batch_size = batch_size,
        gradient_accumulate_every = 1,
        train_lr = lr,
        lr_scheduler = lr_scheduler,
        train_num_steps = train_num_steps,
        save_and_sample_every= save_and_sample_every,
        results_folder=results_folder, 
        amp = False,
        mixed_precision_type = 'bf16',
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