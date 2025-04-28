import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils
from torchvision import utils as tv_utils

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from utils import *
from generate import edm_sampler

# trainer class
class Trainer:
    def __init__(
            self,
            model, 
            loss_fn,
            dataset,
            *,
            train_batch_size = 16,
            gradient_accumulate_every = 1,
            augment_horizontal_flip = True,
            train_lr = 1e-4,
            lr_scheduler = False,
            warmup_fraction = 0.10, 
            train_num_steps = 100000,
            ema_update_every = 10,
            ema_decay = 0.995,
            adam_betas = (0.9, 0.99),
            save_and_sample_every = 1000,
            num_samples = 25,
            results_folder = './results',
            amp = False,
            mixed_precision_type = 'fp16',
            split_batches = True,
            convert_image_to = None,
            calculate_fid = False,
            inception_block_idx = 2048,
            max_grad_norm = 1.,
            num_fid_samples = 10_000,
            save_best_and_latest_only = False,
            num_workers = None
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model
        self.model = model
        self.loss_fn = loss_fn
        self.channels = model.img_channels
        #is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels
        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, \
            f'your effective batch size should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_resolution = model.img_resolution

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        self.ds = dataset
        #self.ds = CustomDataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'
        num_workers = cpu_count() if num_workers is None else num_workers
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = num_workers) #cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        self.opt = Adam(model.parameters(), lr = train_lr, betas = adam_betas)
        if lr_scheduler is not None:
            num_warmup_steps = int(warmup_fraction * train_num_steps) 
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.opt,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=train_num_steps
            )
        else:
            self.lr_scheduler = None

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.loss_fn = self.accelerator.prepare(self.loss_fn)

        
        # # FID-score computation
        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from fid_evaluation import FIDEvaluation

            # if not is_ddim_sampling:
            #     self.accelerator.print(
            #         "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
            #         "Consider using DDIM sampling to save time."
            #     )

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                model=self.ema.ema_model,
                sampling_scheme=edm_sampler,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

            
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        losses = []

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    with self.accelerator.autocast():
                        loss = self.loss_fn(self.model, data).mean()
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')
                losses.append(total_loss)
                #if self.step % 10 == 0: print(self.step, total_loss)
                
                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with self.accelerator.autocast():
                            with torch.inference_mode():
                                milestone = self.step // self.save_and_sample_every
                                batches = num_to_groups(self.num_samples, self.batch_size)
                                #all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                                all_images_list = []
                                for n in batches:
                                    latents = torch.randn(size=(n, self.channels, self.image_resolution, self.image_resolution), device=self.device)
                                    all_images_list.append(edm_sampler(self.ema.ema_model, latents))

                        all_images = torch.cat(all_images_list, dim = 0)

                        tv_utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid
                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')

                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
        return losses
