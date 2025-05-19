import os
import sys
import numpy as np
from pathlib import Path
from multiprocessing import cpu_count

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from torch.optim import Adam

from tqdm.auto import tqdm
from ema_pytorch import EMA

from transformers import get_cosine_schedule_with_warmup
from utils import infinite_dataloader, divisible_by
from interpolant_utils import save_fig, save_losses_fig

typedict = {"fp16":torch.float16, "fp32":torch.float32, "bf16":torch.bfloat16}

# get worker info for distributed training
def get_worker_info():
    if dist.is_initialized():
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
    else:
        print("Distributed training not initialized, using single GPU.")
        rank = 0
        local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return world_size, rank, local_rank, device

# trainer class
class TrainerMLP:
    def __init__(
            self,
            model,
            deconvolver,
            dataset,
            dataset_sampler = None,
            *,
            train_batch_size = 16,
            gradient_accumulate_every = 1,
            train_lr = 1e-4,
            lr_scheduler = False,
            warmup_fraction = 0.10,
            train_num_steps = 100000,
            ema_update_every = 10,
            ema_decay = 0.995,
            adam_betas = (0.9, 0.999),
            save_and_sample_every = 1000,
            results_folder = './results',
            mixed_precision_type = 'fp32',
            max_grad_norm = 1.,
            num_workers = 1,
            clean_data_steps = -1, # not using clean data
            save_fig_fn = save_fig,
            valid_data_plot = None,
    ):
        super().__init__()

        # model
        self.model = model
        self.deconvolver = deconvolver

        self.world_size, self.rank, self.local_rank, self.device = get_worker_info()
        print(self.device)
        self.master_process = self.rank == 0

        # sampling and training hyperparameters
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision_type = mixed_precision_type
        self.clean_data_steps = clean_data_steps
        self.save_fig_fn = save_fig_fn
        self.valid_data_plot = valid_data_plot

        # dataset and dataloader
        if getattr(dataset, '_is_my_custom_distribution_data_loader', False):
            print("Using DistributionDataLoader")
            self.dl = dataset
            self.ds = self.dl.distribution # not used so far
        else:
            print("Using DataLoader")
            self.ds = dataset
            num_workers = cpu_count() if num_workers is None else num_workers
            if dataset_sampler is not None:
                dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True,
                                pin_memory = True, num_workers = 0) #cpu_count())
            else:
                dl = DataLoader(self.ds, batch_size = train_batch_size, sampler = dataset_sampler,
                                pin_memory = True, num_workers = 0)
            self.dl = infinite_dataloader(dl)

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
        if self.master_process:
            self.ema = EMA(model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state
        self.step = 0


    def save(self, milestone):
        if not self.master_process:
            return

        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'ema_model': self.ema.ema_model.state_dict(),
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        device = self.device
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), \
                          map_location=device, weights_only=True)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.master_process:
            self.ema.load_state_dict(data["ema"])
        if 'version' in data:
            print(f"loading from version {data['version']}")


    def train(self):
        device = self.device
        losses = []
        min_loss = 1e10
        if not bool(os.getenv('SLURM_JOB_ID')): # interactive environment like Jupyter
            miniters = 1
            mininterval = 0.1
            pbar_refresh = True
        else: # non-interactive environment
            miniters = 100
            mininterval = 60.0
            pbar_refresh = False
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not self.master_process, miniters=miniters, mininterval=mininterval) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    image, corrupted, latents = next(self.dl)
                    image = image.to(self.device)
                    corrupted = corrupted.to(self.device)
                    latents = latents.to(self.device) if self.deconvolver.use_latents else None
                    with torch.autocast(device_type=device, dtype=typedict[self.mixed_precision_type]):
                        if self.step < self.clean_data_steps:
                            loss = self.deconvolver.loss_fn_cleandata(self.model, corrupted, image, latents).mean()
                        else:
                            loss = self.deconvolver.loss_fn(self.model, corrupted, latents).mean()
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    loss.backward()

                    for p in self.model.parameters():
                        if p.grad is not None and not p.grad.is_contiguous():
                            p.grad = p.grad.contiguous()

                pbar.set_description(f'loss: {total_loss:.4f}', refresh=pbar_refresh)
                losses.append(total_loss)
                #if self.step % 10 == 0: print(self.step, total_loss)

                torch.cuda.synchronize()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                torch.cuda.synchronize()

                self.step += 1
                if self.master_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        if self.step % 5000 == 0:
                            self.save(self.step)
                        self.save("latest")
                        if losses[-1] < min_loss:
                            min_loss = losses[-1]
                            self.save("best")
                            print(f"New best model at step {self.step} with loss {min_loss:.4f}")
                        self.ema.ema_model.eval()

                        with torch.autocast(device_type=device, dtype=typedict[self.mixed_precision_type]):
                            with torch.inference_mode():
                                milestone = self.step // self.save_and_sample_every
                                np.save(f"{self.results_folder}/losses", losses)
                                if self.valid_data_plot is None:
                                    image, corrupted, latents = next(self.dl)
                                    image = image.to(self.device)
                                    corrupted = corrupted.to(self.device)
                                    latents = latents.to(self.device) if self.deconvolver.use_latents else None
                                else:
                                    image, corrupted, latents = self.valid_data_plot
                                clean = self.deconvolver.transport(self.ema.ema_model, corrupted, latents)
                                self.save_fig_fn(milestone, image, corrupted, clean, self.results_folder)
                                print(f"Saved model at step {self.step}")

                pbar.update(1)
                torch.cuda.synchronize()

        # Save final model
        if self.master_process:
            self.save("latest")
            save_losses_fig(losses, self.results_folder)
            self.ema.ema_model.eval()
            with torch.autocast(device_type=device, dtype=typedict[self.mixed_precision_type]):
                with torch.inference_mode():
                    milestone = self.step // self.save_and_sample_every
                    np.save(f"{self.results_folder}/losses", losses)
                    if self.valid_data_plot is None:
                        image, corrupted, latents = next(self.dl)
                        image = image.to(self.device)
                        corrupted = corrupted.to(self.device)
                        latents = latents.to(self.device) if self.deconvolver.use_latents else None
                    else:
                        image, corrupted, latents = self.valid_data_plot
                    clean = self.deconvolver.transport(self.ema.ema_model, corrupted, latents)
                    self.save_fig_fn("fin", image, corrupted, clean, self.results_folder)
                    if self.master_process:
                        print(f"Saved model at step {self.step}")

            print('training complete')

        return losses
