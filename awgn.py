import sys, os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')
import json

sys.path.append('./src/')
from utils import count_parameters, make_serializable
from custom_datasets import dataset_dict, ImagesOnly, CorruptedDataset
from networks import ConditionalDhariwalUNet
from interpolant_utils import DeconvolvingInterpolantAWGN
import forward_maps as fwd_maps
from trainer_si import Trainer
from utils import remove_all_prefix
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
parser.add_argument("--save_every", type=int, default=5000, help="save every steps")
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
parser.add_argument("--save_transport", action='store_true', help="save transport maps on updating")
parser.add_argument("--n_transports", type=int, default=1, help="update transport map every n steps")

args = parser.parse_args()
print(args)

if args.multiview: 
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/multiview/'
else:
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/singleview/'

# Parse arguments
dataset, D, nc = dataset_dict[args.dataset]
dataset = dataset()
if args.dataset == 'celebA':
    image_dataset = dataset
else:
    image_dataset = ImagesOnly(dataset)
model_channels = args.channels #192
train_num_steps = args.train_steps
save_and_sample_every = min(args.save_every, int(train_num_steps//10))
batch_size = args.batch_size
lr = args.learning_rate 
gated = args.gated
if gated: 
    args.suffix = f"{args.suffix}-gated" if args.suffix else "gated"
lr_scheduler = args.lr_scheduler

# Parse corruption arguments
corruption = args.corruption
corruption_levels = args.corruption_levels
noise_scale = float(corruption_levels[0])
print('Noise scale = ', noise_scale)
try:
    fwd_func = fwd_maps.corruption_dict[corruption](*corruption_levels)
except Exception as e:
    print("Exception in loading corruption function : ", e)
    sys.exit()
cname = "-".join([f"{i:0.2f}" for i in corruption_levels])
print(f"Corruption name for levels {corruption_levels}: ", cname)
folder = f"{args.dataset}-{corruption}-{cname}"
if args.cleansteps != -1: folder = f"{folder}-cds{args.cleansteps}"
if args.transport_steps != 1: folder = f"{folder}-tr{args.transport_steps}"
if args.smodel: folder = f"{folder}-sde"
if args.sampler != 'euler': folder = f"{folder}-{args.sampler}"
if args.combinedsde: folder = f"{folder}-combined"
if args.prefix != "": folder = f"{args.prefix}-{folder}"
if args.suffix != "": folder = f"{folder}-{args.suffix}"
folder = f"{folder}-awgn"
results_folder = f"{BASEPATH}/{folder}/"
os.makedirs(results_folder, exist_ok=True)
print(f"Results will be saved in folder: {results_folder}")

use_latents, latent_dim = fwd_maps.parse_latents(corruption, D)
if use_latents:
    print("Will use latents of dimension: ", latent_dim)
args_dict = make_serializable(vars(args) if isinstance(args, argparse.Namespace) else args)
with open(f"{results_folder}/args.json", "w") as f:
    json.dump(args_dict, f, indent=4)

# Initialize model and train
b =  ConditionalDhariwalUNet(D, nc, nc, latent_dim=latent_dim,
                            model_channels=model_channels, gated=gated, \
                            max_pos_embedding=args.max_pos_embedding).to(device)
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

# load model if a path is provided
if args.load_model_path:
    print("Loading model from: ", args.load_model_path)
    from ema_pytorch import EMA
    try:
        data = torch.load(f'{BASEPATH}/{args.load_model_path}', weights_only=True, map_location='cpu')
    except:
        data = torch.load(f'{args.load_model_path}', weights_only=True, map_location='cpu')
    ema = EMA(b)
    ema.load_state_dict(remove_all_prefix(data['ema']))
    b.load_state_dict(ema.ema_model.state_dict())
    if ('s_ema' in data.keys()) and args.smodel:
        print("Loading sde model")
        ema = EMA(s_model)
        ema.load_state_dict(remove_all_prefix(data['s_ema']))
        s_model.load_state_dict(ema.ema_model.state_dict())


#b = torch.compile(b)
print("Parameter count : ", count_parameters(b))
deconvolver = DeconvolvingInterpolantAWGN(fwd_func, use_latents=use_latents, \
                                      alpha=args.alpha, resamples=args.resamples, n_steps=args.ode_steps, \
                                      gamma_scale=args.gamma_scale, diffusion_coeff=args.diffusion_coeff,
                                        sampler=args.sampler, noise_scale=noise_scale).to(device)
corrupt_dataset = CorruptedDataset(image_dataset, deconvolver.push_fwd, \
                                   tied_rng=not(args.multiview), base_seed=args.dataset_seed)

trainer = Trainer(model=b, 
                  deconvolver=deconvolver, 
                  dataset = corrupt_dataset,
                  train_batch_size = batch_size // args.n_transports,
                  gradient_accumulate_every = int(batch_size * args.n_transports// args.mini_batch_size),
                  train_lr = lr,
                  lr_scheduler = lr_scheduler,
                  train_num_steps = train_num_steps,
                  save_and_sample_every= save_and_sample_every,
                  results_folder=results_folder, 
                  warmup_fraction=0.05,
                  update_transport_every=args.transport_steps,
                  s_model=s_model,
                  clean_data_steps=args.cleansteps,
                  save_transport=args.save_transport
        )

trainer.train()
