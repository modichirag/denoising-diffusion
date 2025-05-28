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
from custom_datasets import  CorruptedDataset
from custom_datasets import CombinedNumpyDataset
from networks import ConditionalDhariwalUNet
from interpolant_utils import DeconvolvingInterpolant
from forward_maps import corruption_dict, parse_latents
from trainer_si import Trainer, get_worker_info
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--subsample", type=int, default=1, help="subsample from original data of 320")
parser.add_argument("-D", type=int, default=40, help="effective dimension after shuffling")
parser.add_argument("--corruption", type=str, help="corruption")
parser.add_argument("--corruption_mode", type=str, default='same_rate', help="corruption")
parser.add_argument("--corruption_levels", type=float, nargs='+', help="corruption level")
parser.add_argument("--channels", type=int, default=32, help="number of channels in model")
parser.add_argument("--train_steps", type=int, default=101, help="number of channels in model")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--mini_batch_size", type=int, default=128, help="batch size per iteration")
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
parser.add_argument("--noise_masked", action='store_true', help="add noise to masked region, else not")


args = parser.parse_args()
print(args)

if args.multiview: 
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/multiview/'
else:
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/singleview/'

# Initialize DDP
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
world_size, rank, local_rank, device = get_worker_info()
torch.cuda.set_device(device)
print(f"DEVICE (Rank {local_rank}): {device}")

D0 = 320
sub = args.subsample
D = args.D 
s = int((D0/sub)/D)
print(f"Run for dim : {D0/sub}; with downscaling factor: {s}")
nc = int(s**2)
shuffler = torch.nn.PixelShuffle(s)
unshuffler = torch.nn.PixelUnshuffle(s)
data_folder = f"/mnt/ceph/users/cmodi/ML_data/fastMRI/knee-singlecoil-train-pix-sub{sub}/"
dataset = CombinedNumpyDataset(data_folder, transform = unshuffler)
train_batch_size = int(args.batch_size//world_size)
gradient_accumulate_every = max(1, int(train_batch_size // args.mini_batch_size))


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
corruption = args.corruption
corruption_levels = args.corruption_levels
fwd_func = corruption_dict[corruption](*corruption_levels, \
        mode=args.corruption_mode, downscale_factor=s, noise_masked=args.noise_masked)
use_latents, latent_dim = parse_latents(corruption, D=D, s=s)
if use_latents:
    print("Will use latents of dimension: ", latent_dim)


# Save folder name
cname = "-".join([f"{i:0.2f}" for i in corruption_levels])
folder = f"{corruption}-{cname}"
if args.prefix != "": folder = f"{args.prefix}-{folder}"
if args.suffix != "": folder = f"{folder}-{args.suffix}"
results_folder = f"{BASEPATH}/{folder}/"
os.makedirs(results_folder, exist_ok=True)
print(f"Results will be saved in folder: {results_folder}")
args_dict = make_serializable(vars(args) if isinstance(args, argparse.Namespace) else args)
if local_rank == 0:
    with open(f"{results_folder}/args.json", "w") as f:
        json.dump(args_dict, f, indent=4)


# Setup model
b =  ConditionalDhariwalUNet(D, nc, nc, latent_dim=latent_dim,
                        model_channels=model_channels, gated=gated).to(device)
b = DDP(b, device_ids=[local_rank], find_unused_parameters=False)     
print("Parameter count : ", count_parameters(b))
deconvolver = DeconvolvingInterpolant(fwd_func, use_latents=use_latents, \
                                      alpha=args.alpha, resamples=args.resamples, n_steps=args.ode_steps).to(device)
corrupt_dataset = CorruptedDataset(dataset, deconvolver.push_fwd, \
                                   tied_rng=not(args.multiview), base_seed=args.dataset_seed)
dataset_sampler = DistributedSampler(corrupt_dataset, num_replicas=world_size, \
                                     shuffle=True, rank=local_rank)


print("Launch training")
trainer = Trainer(model=b, 
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
                    warmup_fraction=0.05
        )

trainer.train()

dist.destroy_process_group()
