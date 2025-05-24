import torch
import sys, os
import json
import argparse
from torch.utils.data import DataLoader, Dataset
from ema_pytorch import EMA

sys.path.append('./src/')
from networks import EDMPrecond
from custom_datasets import dataset_dict, ImagesOnly
from fid_evaluation import FIDEvaluation
from utils import cycle
from generate import edm_sampler

BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/'

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="")
parser.add_argument("--folder", type=str, help="Path")
parser.add_argument("--dataset", type=str, help="dataset")
parser.add_argument("--channels", type=int, default=64, help="number of channels in model")
parser.add_argument("--model", type=str, default='best', help="which saved model in folder")
parser.add_argument("--n_samples", type=int, default=50_000, help="Samples to evalaute FID")
parser.add_argument("--batch_size", type=int, default=256, help="batch size")


# Parse arguments
args = parser.parse_args()
print(args)
folder = f"{BASEPATH}/{args.folder}/"
dataset, D, nc = dataset_dict[args.dataset]
image_dataset = ImagesOnly(dataset)
model_channels = args.channels #192
print(D, nc)

#Other parameters
batch_size = args.batch_size
n_samples = args.n_samples
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)


print("Setup model and dataloader")
model = EDMPrecond(D, nc, model_channels=model_channels).to(device)
image_dataset = ImagesOnly(dataset)
dl = DataLoader(image_dataset, batch_size=batch_size, shuffle = True, pin_memory = True, num_workers = 1) 
dl = cycle(dl)
fid_scorer = FIDEvaluation(
    batch_size=batch_size,
    dl=dl,
    channels=nc,
    accelerator=None, #self.accelerator,
    stats_dir=folder,
    device=device,
    num_fid_samples=n_samples,
    inception_block_idx=2048
)
sampling_scheme = lambda net, latents: edm_sampler(net, latents)

#Load model

data = torch.load(f'{folder}/model-{args.model}.pt', map_location=device, weights_only=True)
model.load_state_dict(data['model'])
print("Model loaded")
score = fid_scorer.fid_score(model, sampling_scheme=sampling_scheme, force_calc=True)
print(f"FID score of loaded best model : {score}")

#Load EMA model
ema = EMA(model).to(device)
ema.load_state_dict(data['ema'])
print("Model loaded")
score_ema = fid_scorer.fid_score(ema.ema_model, sampling_scheme=sampling_scheme)
print(f"FID score of corresponding ema model: {score_ema}")

to_save = {'FID_best': score, 'FID_best_ema':score_ema}
n = int(n_samples/1e3)
save_name = f"{folder}/fid_{n}k-{args.model}.json"
with open(save_name, 'w') as file:
        json.dump(to_save, file, indent=4)
