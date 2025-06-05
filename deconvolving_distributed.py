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
from custom_datasets import dataset_dict, ImagesOnly, CorruptedDataset
from networks import ConditionalDhariwalUNet
from interpolant_utils import DeconvolvingInterpolant
import forward_maps as fwd_maps
from trainer_si import Trainer, get_worker_info
from callbacks import save_image
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

parser = argparse.ArgumentParser(description="")
parser.add_argument("--corruption", type=str, help="corruption")
parser.add_argument("--corruption_levels", type=float, nargs='+', help="corruption level")
parser.add_argument("--dataset", type=str, default='cifar10', help="dataset")
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
parser.add_argument("--ode_steps", type=int, default=80, help="ode steps")
parser.add_argument("--alpha", type=float, default=1., help="probability of using new data")
parser.add_argument("--resamples", type=int, default=1, help="number of resamplings")
parser.add_argument("--multiview", action='store_true', help="change corruption every epoch if provided, else not")
parser.add_argument("--max_pos_embedding", type=int, default=2, help="number of resamplings")
parser.add_argument("--transport_steps", type=int, default=1, help="update transport map every n steps")

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


# Parse arguments
dataset, D, nc = dataset_dict[args.dataset]
dataset = dataset()
if args.dataset == 'celebA':
    image_dataset = dataset
else:
    image_dataset = ImagesOnly(dataset)
model_channels = args.channels #192
train_num_steps = args.train_steps
save_and_sample_every = min(500, int(train_num_steps//50))
batch_size = args.batch_size
lr = args.learning_rate 
gated = args.gated
if gated: 
    args.suffix = f"{args.suffix}-gated" if args.suffix else "gated"
lr_scheduler = args.lr_scheduler
train_batch_size = int(batch_size//world_size)
gradient_accumulate_every = max(1, int(train_batch_size // args.mini_batch_size))

# Parse corruption arguments
corruption = args.corruption
corruption_levels = args.corruption_levels
try:
    fwd_func = fwd_maps.corruption_dict[corruption](*corruption_levels)
except Exception as e:
    print("Exception in loading corruption function : ", e)
    sys.exit()
use_latents, latent_dim = fwd_maps.parse_latents(corruption, D)
if use_latents:
    print("Will use latents of dimension: ", latent_dim)

# Folder name
cname = "-".join([f"{i:0.2f}" for i in corruption_levels])
print(f"Corruption name for levels {corruption_levels}: ", cname)
folder = f"{args.dataset}-{corruption}-{cname}"
if args.prefix != "": folder = f"{args.prefix}-{folder}"
if args.suffix != "": folder = f"{folder}-{args.suffix}"
results_folder = f"{BASEPATH}/{folder}/"
os.makedirs(results_folder, exist_ok=True)
print(f"Results will be saved in folder: {results_folder}")
args_dict = make_serializable(vars(args) if isinstance(args, argparse.Namespace) else args)
if local_rank == 0:
    with open(f"{results_folder}/args.json", "w") as f:
        json.dump(args_dict, f, indent=4)

# Initialize model and train
b =  ConditionalDhariwalUNet(D, nc, nc, latent_dim=latent_dim,
                            model_channels=model_channels, gated=gated, \
                            max_pos_embedding=args.max_pos_embedding).to(device)
# b = DDP(b, device_ids=[local_rank], find_unused_parameters=False)     
print("Parameter count : ", count_parameters(b))
deconvolver = DeconvolvingInterpolant(fwd_func, use_latents=use_latents, \
                                    alpha=args.alpha, resamples=args.resamples, \
                                    n_steps=args.ode_steps).to(device)
corrupt_dataset = CorruptedDataset(image_dataset, deconvolver.push_fwd, \
                                   tied_rng=not(args.multiview), base_seed=args.dataset_seed)
dataset_sampler = DistributedSampler(corrupt_dataset, num_replicas=world_size, \
                                     shuffle=True, rank=local_rank)

print("Launch Train")
trainer = Trainer(model=b, 
                ddp = ddp,
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
                update_transport_steps=args.transport_steps,
                callback_fn =save_image,
        # mixed_precision_type = 'fp32',
        )

trainer.train()

dist.destroy_process_group()