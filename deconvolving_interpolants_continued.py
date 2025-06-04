import sys, os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from ema_pytorch import EMA
from torch.optim import Adam

sys.path.append('./src/')
from utils import grab, cycle, count_parameters, infinite_dataloader
from custom_datasets import dataset_dict, ImagesOnly, CorruptedDataset
from networks import ConditionalDhariwalUNet
from interpolant_utils import DeconvolvingInterpolant
import forward_maps as fwd_maps
from trainer_si import Trainer
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--model_path", type=str, default='latest', help="which model to load")
parser.add_argument("--resume_count", type=int, default=1, help="continued training count")
#standard arguments
parser.add_argument("--corruption", type=str, help="corruption")
parser.add_argument("--dataset", type=str, default='cifar10', help="dataset")
parser.add_argument("--corruption_levels", type=float, nargs='+', help="corruption level")
parser.add_argument("--channels", type=int, default=32, help="number of channels in model")
parser.add_argument("--train_steps", type=int, default=101, help="number of channels in model")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
parser.add_argument("--prefix", type=str, default='', help="prefix for folder name")
parser.add_argument("--suffix", type=str, default='', help="suffix for folder name")
parser.add_argument("--gated", action='store_true', help="gated convolution if provided, else not")
parser.add_argument("--lr_scheduler", action='store_true', help="use scheduler if provided, else not")
parser.add_argument("--dataset_seed", type=int, default=42, help="corrupt dataset seed")
parser.add_argument("--ode_steps", type=int, default=80, help="ode steps")
parser.add_argument("--multiview", action='store_true', help="change corruption every epoch if provided, else not")
parser.add_argument("--alpha", type=float, default=1., help="probability of using new data")
parser.add_argument("--resamples", type=int, default=1, help="number of resamplings")
parser.add_argument("--n_saves", type=int, default=50, help="how frequent to save")
args = parser.parse_args()
print(args)

if args.multiview: 
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/multiview/'
else:
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/singleview/'

# Parse arguments
dataset, D, nc = dataset_dict[args.dataset]
dataset = dataset()
image_dataset = ImagesOnly(dataset)
model_channels = args.channels #192
train_num_steps = args.train_steps
save_and_sample_every = min(200, int(train_num_steps//args.n_saves))
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
folder = f"{BASEPATH}/{folder}/"
model_path = f'{folder}/{args.model_path}'
assert os.path.exists(model_path), f"Folder {folder} does not exist"
results_folder = f"{folder}/resume{args.resume_count}/"
os.makedirs(results_folder, exist_ok=True)
print(f"Results will be saved in folder: {results_folder}")

# Initialize model, load it, and train
use_latents, latent_dim = fwd_maps.parse_latents(corruption, D)
if use_latents:
    print("Will use latents of dimension: ", latent_dim)
b =  ConditionalDhariwalUNet(D, nc, nc, latent_dim=latent_dim,
                        model_channels=model_channels, gated=gated).to(device)
print("Parameter count : ", count_parameters(b))

# data = torch.load(f'{model_path}', weights_only=True)
# b.load_state_dict(data['model'])
# b.to(device)
# ema_b = EMA(b)
# ema_b.load_state_dict(data['ema'])
# b.load_state_dict(ema_b.ema_model.state_dict())
# del ema_b
# print("model loaded")
# opt = Adam(b.parameters(), lr = args.learning_rate, betas = (0.9, 0.999))
# opt.load_state_dict(data['opt'])
# print("loaded learning rate : ", opt.param_groups[0]['lr'])
# for param_group in opt.param_groups:
#     param_group['lr'] = args.learning_rate
# print("learning rate reset to: ", opt.param_groups[0]['lr'])

deconvolver = DeconvolvingInterpolant(fwd_func, use_latents=use_latents, \
                                      alpha=args.alpha, resamples=args.resamples, n_steps=args.ode_steps).to(device)
corrupt_dataset = CorruptedDataset(image_dataset, deconvolver.push_fwd, \
                                   tied_rng=not(args.multiview), base_seed=args.dataset_seed)
trainer = Trainer(model=b, 
        deconvolver=deconvolver, 
        #optimizer = opt,
        dataset = corrupt_dataset,
        train_batch_size = batch_size,
        gradient_accumulate_every = 1,
        train_lr = lr,
        lr_scheduler = lr_scheduler,
        train_num_steps = train_num_steps,
        save_and_sample_every= save_and_sample_every,
        results_folder=results_folder, 
        milestone=model_path, 
        )

trainer.train()
