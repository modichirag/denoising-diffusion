import numpy as np
import torch
import  matplotlib.pyplot as plt
import sys, os
sys.path.append('./src/')
from networks import EDMPrecond
from custom_datasets import cifar10_train, ImagesOnly
from torch.utils.data import DataLoader, Dataset
from fid_evaluation import FIDEvaluation
from ema_pytorch import EMA
from utils import cycle


folder = str(sys.argv[1])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE : ", device)

D = 32
nc = 3
model_channels = 96 #192
batch_size = 256
n_samples = 50_000


print("Setup model and dataloader")
model = EDMPrecond(D, nc, model_channels=model_channels).to(device)
cifar10_train_images = ImagesOnly(cifar10_train)
dl = DataLoader(cifar10_train_images, batch_size=batch_size, shuffle = True, pin_memory = True, num_workers = 1) 
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

# score = fid_scorer.fid_score(model, force_calc=True)
# print(f"FID score of init model : {score}")

#Load model
data = torch.load(f'{folder}/model-best.pt', map_location=device, weights_only=True)
model.load_state_dict(data['model'])
print("Model loaded")
score2 = fid_scorer.fid_score(model, force_calc=True)
print(f"FID score of clean loaded model : {score2}")

#Load EMA model
ema = EMA(model).to(device)
ema.load_state_dict(data['ema'])
print("Model loaded")
score2_ema = fid_scorer.fid_score(ema.ema_model)
print(f"FID score of ema model: {score2_ema}")
