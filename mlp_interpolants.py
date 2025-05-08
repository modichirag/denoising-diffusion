import sys, os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

sys.path.append('./src/')
from utils import  cycle, count_parameters, infinite_dataloader, grab
from nets import SimpleFeedForward
from distribution import DistributionDataLoader, CheckerDistribution
from interpolant_utils import VelocityField, DeconvolvingInterpolant, save_fig_checker
import forward_maps as fwd_maps
from trainer_si import Trainer
import argparse


BASEPATH = '/mnt/home/jhan/stoch-int-priors/results'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--dataset", type=str, default="checker", help="dataset")
parser.add_argument("--corruption", type=str, default="gaussian_noise", help="corruption")
parser.add_argument("--corruption_levels", type=float, nargs='+', help="corruption level")
parser.add_argument("--fc_width", type=int, default=128, help="width of the feedforward network")
parser.add_argument("--fc_depth", type=int, default=3, help="depth of the feedforward network")
parser.add_argument("--train_steps", type=int, default=10000, help="number of channels in model")
parser.add_argument("--batch_size", type=int, default=4000, help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--prefix", type=str, default='', help="prefix for folder name")
parser.add_argument("--suffix", type=str, default='', help="suffix for folder name")
parser.add_argument("--lr_scheduler", action='store_true', help="use scheduler if provided, else not")

# Parse arguments
args = parser.parse_args()
# args = parser.parse_args(['--corruption_levels', '0.4'])

print(args)
train_num_steps = args.train_steps
save_and_sample_every = int(train_num_steps//10)
batch_size = args.batch_size
lr = args.learning_rate
lr_scheduler = args.lr_scheduler

# Parse corruption arguments
corruption = args.corruption # to fix
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
use_latents, latent_dim = fwd_maps.parse_latents(corruption, None)

# Initialize model and train
b =  SimpleFeedForward(2, [args.fc_width]*args.fc_depth).to(device)
print("Parameter count : ", count_parameters(b))
dl = DistributionDataLoader(CheckerDistribution(device=device), batch_size=batch_size)
deconvolver = DeconvolvingInterpolant(fwd_func, use_latents=use_latents, n_steps=40).to(device)

# trainer = Trainer(model=b,
#         deconvolver=deconvolver,
#         dataset = image_dataset,
#         train_batch_size = batch_size,
#         gradient_accumulate_every = 1,
#         train_lr = lr,
#         lr_scheduler = lr_scheduler,
#         train_num_steps = train_num_steps,
#         save_and_sample_every= save_and_sample_every,
#         results_folder=results_folder,
#         )

# trainer.train()

def train_step(xn, b, deconvolver, opt, sched):
    # ts  = torch.rand(xn.shape[0]).to(device)
    loss_val = deconvolver.loss_fn(b, xn)
    # perform backprop
    loss_val.backward()
    opt.step()
    sched.step()
    opt.zero_grad()
    res = {'loss': loss_val.detach().cpu()}
    return res

opt = torch.optim.Adam([{'params': b.parameters(), 'lr': lr} ])
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.9)

losses = []
pbar = tqdm(range(train_num_steps))
for i in pbar:
    b.train()
    x = next(dl).to(device)
    xn = deconvolver.push_fwd(x)
    res = train_step(xn, b, deconvolver, opt, sched)
    loss = res['loss'].detach().numpy().mean()
    losses.append(loss)
    pbar.set_description(f'Loss: {loss:.4f}')
    if i % save_and_sample_every == 0:
        torch.save(b.state_dict(), f"{results_folder}/model.pt")
        np.save(f"{results_folder}/losses", losses)
        clean = dl.distribution.sample(20000).to(device)
        corrupted = deconvolver.push_fwd(clean)
        generated = deconvolver.transport(b, corrupted)
        save_fig_checker(i, grab(clean), grab(corrupted), grab(generated), results_folder, None)
        print(f"Saved model at step {i}")

torch.save(b.state_dict(), f"{results_folder}/model.pt")
np.save(f"{results_folder}/losses", losses)
clean = dl.distribution.sample(20000).to(device)
corrupted = deconvolver.push_fwd(clean)
generated = deconvolver.transport(b, corrupted)
save_fig_checker('final', grab(clean), grab(corrupted), grab(generated), results_folder, None)