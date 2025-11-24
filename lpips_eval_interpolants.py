import torch
import sys, os
import json
import argparse
from torch.utils.data import DataLoader, Dataset
from ema_pytorch import EMA
import numpy as np

sys.path.append('./src/')
from networks import ConditionalDhariwalUNet
from custom_datasets import dataset_dict, ImagesOnly, cifar10_inverse_transforms
from interpolant_utils import DeconvolvingInterpolant
import forward_maps as fwd_maps
import lpips
from utils import infinite_dataloader,  num_to_groups, remove_orig_mod_prefix
from tqdm.auto import tqdm
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="")
parser.add_argument("--model", type=str, default="best", help="which model to load")
parser.add_argument("--dataset", type=str, help="dataset")
parser.add_argument("--corruption", type=str, help="corruption")
parser.add_argument("--corruption_levels", type=float, nargs='+', help="corruption level")
parser.add_argument("--channels", type=int, default=64, help="number of channels in model")
parser.add_argument("--n_samples", type=int, default=50_000, help="Samples to evalaute FID")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--prefix", type=str, default='', help="prefix for folder name")
parser.add_argument("--suffix", type=str, default='', help="suffix for folder name")
parser.add_argument("--subfolder", type=str, default='', help="subfolder for folder name")
parser.add_argument("--gated", action='store_true', help="gated convolution if provided, else not")
parser.add_argument("--ode_steps", type=int, default=80, help="number of steps for ODE sampling")
parser.add_argument("--multiview", action='store_true', help="change corruption every epoch if provided, else not")
parser.add_argument("--max_pos_embedding", type=int, default=2, help="number of resamplings")
args = parser.parse_args()
print(args)
if args.multiview:
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/multiview/'
else:
    BASEPATH = '/mnt/ceph/users/cmodi/diffusion_guidance/singleview/'

# Parse arguments
dataset, D, nc = dataset_dict[args.dataset]
dataset = dataset()
dl = infinite_dataloader(DataLoader(ImagesOnly(dataset), 
                                    batch_size = args.batch_size, \
                                    shuffle = True, pin_memory = True, num_workers = 1))
gated = args.gated
if gated: 
    args.suffix = f"{args.suffix}-gated" if args.suffix else "gated"
print(BASEPATH)

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
if args.subfolder != "": folder = f"{folder}/{args.subfolder}/"

folder = f"{BASEPATH}/{folder}/"
results_folder = f"{folder}/results"
os.makedirs(results_folder, exist_ok=True)
print(f"Models will be loaded from folder: {folder}")
use_latents, latent_dim = fwd_maps.parse_latents(corruption, D)
if use_latents:
    print("Will use latents of dimension: ", latent_dim)
n = int(args.n_samples/1e3)
save_name = f"{results_folder}/lpips_{n}k_{args.ode_steps}steps_{args.model}"
print(f"Results will be saved in file: {save_name}")


deconvolver = DeconvolvingInterpolant(fwd_func, use_latents=use_latents, n_steps=args.ode_steps).to(device)

for emb in [True, False]:
    try:
        b = ConditionalDhariwalUNet(D, nc, nc, latent_dim=latent_dim, model_channels=args.channels, gated=gated, \
                                    max_pos_embedding=args.max_pos_embedding, zero_emb_channels_bwd=emb).to(device)
        ema_b = EMA(b)
        data = torch.load(f'{folder}/model-{args.model}.pt', weights_only=True)
        cleaned_ckpt = remove_orig_mod_prefix(data['model'])
        b.load_state_dict(cleaned_ckpt)
        cleaned_ckpt = remove_orig_mod_prefix(data['ema'])
        ema_b.load_state_dict(cleaned_ckpt)    
        b = ema_b.ema_model
        continue 
    except Exception as e:
        print(e)
     
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
psnr = PeakSignalNoiseRatio(data_range=1.0, dim=[1, 2, 3], reduction=None).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction=None).to(device)

@torch.inference_mode()
def get_cleaned_samples():
    image = next(dl).to(device)
    corrupted, latents = deconvolver.push_fwd(image, return_latents=True)
    latents = latents if use_latents else None
    clean = deconvolver.transport(b, corrupted, latents)
    return image, clean



batches = num_to_groups(args.n_samples, args.batch_size)
loss_alex, loss_vgg = [], []
psnr_list, ssim_list = [], []
for batch in tqdm(batches):
    images, cleaned_images = get_cleaned_samples()    
    with torch.no_grad():
        img1 = torch.clip(cifar10_inverse_transforms(images)*2-1, -1, 1)
        img2 = torch.clip(cifar10_inverse_transforms(cleaned_images)*2-1, -1, 1)
        l1 = (loss_fn_alex(img1.to('cpu'), img2.to('cpu')).detach()).numpy()
        l2 = (loss_fn_vgg(img1.to('cpu'), img2.to('cpu')).detach()).numpy()
        loss_alex.append(l1)
        loss_vgg.append(l2)
        target = torch.clip(cifar10_inverse_transforms(images), 0, 1)
        preds = torch.clip(cifar10_inverse_transforms(cleaned_images), 0, 1)
        psnr_value = psnr(preds, target).to('cpu').numpy()
        ssim_value = ssim(preds, target).to('cpu').numpy()  
        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)

losses1 = np.squeeze(np.concatenate(loss_alex))
losses2 = np.squeeze(np.concatenate(loss_vgg))
stacked_losses = np.stack([losses1, losses2], axis=1)
losses = stacked_losses.mean(axis=0).tolist()
print(f"LPIPS losses : {losses}")
np.save(save_name, stacked_losses)

psnr = np.squeeze(np.concatenate(psnr_list))
ssim = np.squeeze(np.concatenate(ssim_list))
save_name = f"{results_folder}/psnr_{n}k_{args.ode_steps}steps_{args.model}"
np.save(save_name, psnr)
save_name = f"{results_folder}/ssim_{n}k_{args.ode_steps}steps_{args.model}"
np.save(save_name, ssim)
print(f"PSNR/SSIM : {psnr.mean()}/{ssim.mean()}")

to_save = {'LPIPS_alex': losses[0],
           'LPIPS_vgg': losses[1],
           'PSNR': float(psnr.mean()),
           'SSIM':float(ssim.mean())
}
save_name = f"{results_folder}/metrics_{n}k_{args.ode_steps}steps_{args.model}"
with open(save_name + '.json', 'w') as file:
        json.dump(to_save, file, indent=4)



# try:
#     b.load_state_dict(data['model'])
#     ema_b.load_state_dict(data['ema'])
# except Exception as e :
#     print("Saved compiled model. Trying to load without compilation")
#     cleaned_ckpt = remove_orig_mod_prefix(data['model'])
#     b.load_state_dict(cleaned_ckpt)
#     cleaned_ckpt = remove_orig_mod_prefix(data['ema'])
#     ema_b.load_state_dict(cleaned_ckpt)    
# b = ema_b.ema_model

