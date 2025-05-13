import sys, os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

sys.path.append('./src/')
from utils import  cycle, count_parameters, infinite_dataloader, grab
from nets import SimpleFeedForward, FeedForwardwithEMB
from custom_datasets import ManifoldDataset
from distribution import DistributionDataLoader, distribution_dict
from interpolant_utils import DeconvolvingInterpolant, save_fig_checker, save_fig_manifold
import forward_maps as fwd_maps
from trainer_si import Trainer
import argparse
import matplotlib.pyplot as plt


BASEPATH = '/mnt/home/jhan/stoch-int-priors/results'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--dataset", type=str, default="moon", help="dataset")
parser.add_argument("--corruption", type=str, default="projection_vec", help="corruption")
parser.add_argument("--corruption_levels", type=float, nargs='+', help="corruption level")
parser.add_argument("--fc_width", type=int, default=256, help="width of the feedforward network")
parser.add_argument("--fc_depth", type=int, default=3, help="depth of the feedforward network")
parser.add_argument("--train_steps", type=int, default=10000, help="number of channels in model")
parser.add_argument("--batch_size", type=int, default=4000, help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
parser.add_argument("--prefix", type=str, default='', help="prefix for folder name")
parser.add_argument("--suffix", type=str, default='', help="suffix for folder name")
parser.add_argument("--lr_scheduler", action='store_true', help="use scheduler if provided, else not")

# Parse arguments
args = parser.parse_args()
# args = parser.parse_args(['--corruption_levels', '2.0', '0.05',
#                           '--suffix', 'test'])

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
clean_data_step = -2000
alpha = 1.0
use_follmer = False
if use_follmer:
    diffusion_coef = corruption_levels[1]
else:
    diffusion_coef = None
deconvolver = DeconvolvingInterpolant(fwd_func, use_latents=use_latents, n_steps=40, alpha=alpha, diffusion_coef=diffusion_coef).to(device)
if use_follmer:
    deconvolver.transport = deconvolver.transport_follmer
    deconvolver.loss_fn = deconvolver.loss_fn_follmer
    deconvolver.loss_fn_cleandata = deconvolver.loss_fn_follmer_cleandata
if args.dataset in ["checker", "moon"]:
    dim_in = 2
    dl = DistributionDataLoader(distribution_dict[args.dataset](device=device), batch_size=batch_size)
    save_fig_fn = save_fig_checker
    clean_data_valid = dl.distribution.sample(20000).to(device)
elif args.dataset == "manifold_ds":
    dim_in = 5
    dataset = ManifoldDataset("/mnt/home/jhan/diffusion-priors/experiments/manifold/manifold_dataset.npz", epsilon=0.01)
    dl = infinite_dataloader(DataLoader(dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 1))
    save_fig_fn = save_fig_manifold
    clean_data_valid = dataset.x_data[:5000].to(device)
corrupted_valid, latents_valid = deconvolver.push_fwd(clean_data_valid, return_latents=True)
latents_valid = latents_valid if use_latents else None
if args.corruption == "projection_coeff":
    latent_dim = dim_in * int(args.corruption_levels[0])
else:
    latent_dim = None
if args.corruption == "projection_coeff" and dim_in == int(args.corruption_levels[0]):
    corrupted_valid_plot = torch.linalg.solve(latents_valid, corrupted_valid)
else:
    corrupted_valid_plot = corrupted_valid

# to update architecture
# b =  SimpleFeedForward(dim_in, [args.fc_width]*args.fc_depth, latent_dim=latent_dim, use_follmer=use_follmer).to(device)
b =  FeedForwardwithEMB(dim_in, 64, [args.fc_width]*args.fc_depth, latent_dim=latent_dim, use_follmer=use_follmer).to(device)
print("Parameter count : ", count_parameters(b))

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

def train_step(x, xn, b, latents, deconvolver, opt, sched, use_cleandata=False):
    if use_cleandata:
        loss_val = deconvolver.loss_fn_cleandata(b, xn, x, latents)
    else:
        loss_val = deconvolver.loss_fn(b, xn, latents)
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
    # TODO: update dl for checker
    if args.dataset in ["checker", "moon"]:
        x = next(dl).to(device)
        xn, latents = deconvolver.push_fwd(x, return_latents=True)
        latents = latents if use_latents else None
    else:
        x, xn, latents = next(dl)
        x = x.to(device)
        xn = xn.to(device)
        latents = latents.to(device) if deconvolver.use_latents else None
    res = train_step(x, xn, b, latents, deconvolver, opt, sched, use_cleandata=i<clean_data_step)
    loss = res['loss'].detach().numpy().mean()
    losses.append(loss)
    pbar.set_description(f'Loss: {loss:.4f}')
    if i % save_and_sample_every == 0:
        torch.save(b.state_dict(), f"{results_folder}/model.pt")
        np.save(f"{results_folder}/losses", losses)
        generated = deconvolver.transport(b, corrupted_valid, latents_valid)
        save_fig_fn(i, grab(clean_data_valid), grab(corrupted_valid_plot), grab(generated), results_folder, None)
        print(f"Saved model at step {i}")

torch.save(b.state_dict(), f"{results_folder}/model.pt")
np.save(f"{results_folder}/losses", losses)
generated = deconvolver.transport(b, corrupted_valid, latents_valid)
save_fig_fn('final', grab(clean_data_valid), grab(corrupted_valid_plot), grab(generated), results_folder, None)

# Plotting the loss curve
# losses = np.load(f"{results_folder}/losses.npy")
steps = np.arange(len(losses))
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].semilogy(steps, losses, marker='.', linestyle='-', markersize=4, alpha=0.7)
axs[0].set_xlabel("Steps")
axs[0].set_ylabel("Loss (log scale)")
axs[0].set_title("Loss Curve (Semi-Log Y Scale)")
axs[0].grid(True, which="both", ls="--", alpha=0.5)
axs[1].loglog(steps, losses, marker='.', linestyle='-', markersize=4, alpha=0.7, color='orangered')
axs[1].set_xlabel("Steps (log scale)")
axs[1].set_ylabel("Loss (log scale)")
axs[1].set_title("Loss Curve (Log-Log Scale)")
axs[1].grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, 'losses.png'), dpi=300, bbox_inches='tight')
plt.show()
