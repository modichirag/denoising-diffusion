import sys, os
import torch
import torch.distributed as dist
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader

sys.path.append('./src/')
from utils import count_parameters, infinite_dataloader
from custom_datasets import dataset_dict, ImagesOnly
from networks import ConditionalDhariwalUNet
from interpolant_utils import DeconvolvingInterpolant
import forward_maps as fwd_maps
from trainer_si2 import Trainer, get_worker_info
import argparse

BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/'

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
parser.add_argument("--scheduler", action='store_true', help="use scheduler if provided, else not")

# Initialize DDP
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
world_size, rank, local_rank, device = get_worker_info()
torch.cuda.set_device(device)
print(f"DEVICE (Rank {local_rank}): {device}")

# Parse arguments
args = parser.parse_args()
print(args)
model_channels = args.channels #192
train_num_steps = args.train_steps
save_and_sample_every = int(train_num_steps//50)
batch_size = args.batch_size
lr = args.learning_rate 
gated = args.gated
lr_scheduler = args.scheduler
# Dataset arguments
dataset, D, nc = dataset_dict[args.dataset]
image_dataset = ImagesOnly(dataset)
dataset_sampler = DistributedSampler(image_dataset, num_replicas=world_size, \
                                     shuffle=True, rank=local_rank)
# Parse corruption arguments
corruption = args.corruption
corruption_levels = args.corruption_levels
try:
    fwd_func = fwd_maps.corruption_dict[corruption](*corruption_levels)
except Exception as e:
    print(e)
    sys.exit()

# Make folder
cname = "-".join([f"{i:0.2f}" for i in corruption_levels])
folder = f"{args.dataset}-{corruption}-{cname}"
if args.prefix != "": folder = f"{args.prefix}-{folder}"
if args.suffix != "": folder = f"{folder}-{args.suffix}"
results_folder = f"{BASEPATH}/{folder}/"
os.makedirs(results_folder, exist_ok=True)
print(f"Results will be saved in folder: {results_folder}")
use_latents = True if 'mask' in corruption else False
print("Will be using latents: ", use_latents)

# Initialize model and train
b =  ConditionalDhariwalUNet(D, nc, nc, model_channels=model_channels, gated=gated).to(device)
# b = VelocityField(model, use_compile=True)
b = DDP(b, device_ids=[local_rank])  # Wrap model with DDP
print("Parameter count : ", count_parameters(b))
deconvolver = DeconvolvingInterpolant(fwd_func, use_latents=use_latents, n_steps=80).to(device)

trainer = Trainer(model=b, 
        deconvolver=deconvolver, 
        dataset = image_dataset,
        dataset_sampler=dataset_sampler,
        train_batch_size = batch_size//world_size,
        gradient_accumulate_every = 1,
        train_lr = lr,
        lr_scheduler = lr_scheduler,
        train_num_steps = train_num_steps,
        save_and_sample_every= save_and_sample_every,
        results_folder=results_folder, 
        mixed_precision_type = 'fp32',
        )

trainer.train()

dist.destroy_process_group()


# def train_step(xn, b, deconvolver, opt, sched):
#     # with torch.autocast(device_type=device, dtype=dtype):
#     loss_val = deconvolver.loss_fn(b, xn)  
#     loss_val.backward()
#     opt.step()
#     sched.step()
#     opt.zero_grad()    
#     res = {'loss': loss_val.detach().cpu()}
#     return res

# # opt = torch.optim.Adam(b.parameters(), lr=lr)
# # sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.9)
# # dl = infinite_dataloader(DataLoader(image_dataset, batch_size = batch_size, 
# #                         sampler=dataset_sampler, pin_memory = True, num_workers = 1))
# # x = next(dl).to(device)
# # if local_rank == 0: print("First run")
# # xn = deconvolver.push_fwd(x)
# # res = train_step(xn, b, deconvolver, opt, sched)
# # print(res)
