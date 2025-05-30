import os
import numpy as np
from pathlib import Path
from multiprocessing import cpu_count

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from torch.optim import Adam, AdamW
from tqdm.auto import tqdm
from ema_pytorch import EMA

from transformers import get_cosine_schedule_with_warmup
from utils import infinite_dataloader, divisible_by, push_to_device
from callbacks import save_losses_fig

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
class Trainer:
    def __init__(
            self,
            model,
            deconvolver,
            optimizer = None,
            dataset = None,
            dataloader = None,
            dataset_sampler = None,
            *,
            train_batch_size = 16,
            gradient_accumulate_every = 1,
            update_transport_every = 1,
            train_lr = 1e-4,
            lr_scheduler = False,
            warmup_fraction = 0.10,
            train_num_steps = 100000,
            ema_update_every = 10,
            ema_decay = 0.995,
            adam_betas = (0.9, 0.999),
            weight_decay = 0.00,
            save_and_sample_every = 1000,
            results_folder = './results',
            mixed_precision_type = 'fp32',
            max_grad_norm = 1.,
            num_workers = 1,
            milestone = None,
            clean_data_steps = -1, # not using clean data
            callback_fn = None,
            validation_data = None
    ):
        super().__init__()

        # model
        self.model = model
        self.deconvolver = deconvolver

        self.world_size, self.rank, self.local_rank, self.device = get_worker_info()
        self.master_process = self.rank == 0

        # sampling and training hyperparameters
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision_type = mixed_precision_type
        self.update_transport_every = update_transport_every
        self.clean_data_steps = clean_data_steps
        self.callback_fn = callback_fn
        self.validation_data = validation_data

        # dataset and dataloader
        if (dataset is None) and (dataloader is None):
            raise ValueError("Either dataset or dataloader must be provided.")
        if dataloader is not None:
            self.dl = dataloader
        else:
            self.ds = dataset
            num_workers = cpu_count() if num_workers is None else num_workers
            if dataset_sampler is not None:
                dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True,
                                pin_memory = True, num_workers = num_workers) #cpu_count())
            else:
                dl = DataLoader(self.ds, batch_size = train_batch_size, sampler = dataset_sampler,
                                pin_memory = True, num_workers = num_workers)
            self.dl = infinite_dataloader(dl)

        # optimizer
        if optimizer is not None:
            self.opt = optimizer
        else:
            if weight_decay == 0:
                self.opt = Adam(model.parameters(), lr = train_lr, betas = adam_betas)
            else:
                self.opt = AdamW(model.parameters(), lr = train_lr, betas = adam_betas, weight_decay=weight_decay)
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
        if milestone is not None:
            print("Milestone is not None. Will be loading model")
            self.load(milestone)


    def save(self, milestone):
        if not self.master_process:
            return
        data = {
            'step': self.step,
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
        }
        if hasattr(self.model, "_orig_mod"):
            data['model'] = self.model._orig_mod.state_dict()
        elif isinstance(self.model, DDP):
            data['model'] = self.model.module.state_dict()
        else:
            data['model'] = self.model.state_dict()

        if hasattr(self.ema.ema_model, "_orig_mod"):
            data['ema_model'] = self.ema.ema_model._orig_mod.state_dict()
        elif isinstance(self.ema.ema_model, DDP):
            data['ema_model'] = self.ema.ema_model.module.state_dict()
        else:
            data['ema_model'] = self.ema.ema_model.state_dict()

        if self.lr_scheduler is not None:
            data['scheduler'] = self.lr_scheduler.state_dict()

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))


    def load(self, milestone):
        device = self.device
        try:
            data = torch.load(milestone, \
                          map_location=device, weights_only=True)
            print(f"Loading model from {milestone}")
        except Exception as e:
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), \
                          map_location=device, weights_only=True)
            print(f"Loading model from {str(self.results_folder / f'model-{milestone}.pt')}")
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if ('scheduler' in data.keys()) & (self.lr_scheduler is not None):
            self.lr_scheduler.load_state_dict(data['scheduler'])

        if self.master_process:
            self.ema.load_state_dict(data["ema"])
        if 'version' in data:
            print(f"loading from version {data['version']}")
        print("Successfully loaded model")


    def train(self, loss_threshold=10.0, window=11):
        device = self.device
        losses = []
        min_loss = 1e10
        recent_losses = []
        import copy
        if self.update_transport_every > 1:
            print(f"Setting up transport map to be updated every {self.update_transport_every} steps")
            transport_map = copy.deepcopy(self.model.module) if isinstance(self.model, DDP) \
                        else copy.deepcopy(self.model)
            transport_map.eval()
        else:
            transport_map = None

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not self.master_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data, obs, latents = next(self.dl)
                    data, obs = push_to_device(data, obs, device=device)
                    latents = latents.to(self.device) if self.deconvolver.use_latents else None
                    with torch.autocast(device_type=device, dtype=typedict[self.mixed_precision_type]):
                        if self.step < self.clean_data_steps:
                            loss = self.deconvolver.loss_fn(self.model, obs, latents, x0=data).mean()
                        else:
                            loss = self.deconvolver.loss_fn(self.model, obs, latents, b=transport_map).mean()
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    loss.backward()

                    for p in self.model.parameters():
                        if p.grad is not None and not p.grad.is_contiguous():
                            p.grad = p.grad.contiguous()

                pbar.set_description(f'loss: {total_loss:.4f}')
                losses.append(total_loss)

                torch.cuda.synchronize()
                _ = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                torch.cuda.synchronize()

                # If loss spikes, reset model and optimizer
                reset_model = False
                recent_losses.append(total_loss)
                if len(recent_losses) > window:
                    recent_losses.pop(0)
                if len(recent_losses) == window:
                    mean_loss = sum(recent_losses[:-1]) / (window-1)
                    if total_loss > loss_threshold * mean_loss or not torch.isfinite(loss):
                        print("Loss spike detected. Resetting model and optimizer.")
                        self.load("best")
                        recent_losses.clear()
                        self.opt.zero_grad()
                        reset_model = True

                if not reset_model:
                    self.step += 1

                    # Update transport map if needed
                    if (self.step % self.update_transport_every == 0) & (transport_map is not None):
                        if isinstance(self.ema.ema_model, DDP) :
                            transport_map.load_state_dict(self.ema.ema_model.module.state_dict())
                        else:
                            transport_map.load_state_dict(self.ema.ema_model.state_dict())
                        transport_map.eval()

                    if self.master_process:
                        self.ema.update()

                        if self.step % 5000 == 0:
                            self.save(self.step)

                        if divisible_by(self.step, self.save_and_sample_every):
                            np.save(f"{self.results_folder}/losses", losses)
                            self.save("latest")
                            if losses[-1] < min_loss:
                                min_loss = losses[-1]
                                self.save("best")
                                print(f"New best model at step {self.step} with loss {min_loss:.4f}")

                            self.ema.ema_model.eval()
                            model_to_use = self.ema.ema_model.module if isinstance(self.ema.ema_model, DDP) \
                                                else self.ema.ema_model
                            try:
                                with torch.no_grad(), torch.autocast(device_type=device, dtype=typedict[self.mixed_precision_type]):
                                    if self.callback_fn is not None:
                                        milestone=self.step // self.save_and_sample_every
                                        self.callback_fn(idx = milestone,
                                                        b = model_to_use, deconvolver = self.deconvolver,
                                                        dataloader = self.dl, validation_data = self.validation_data,
                                                        losses = losses, device = self.device,
                                                        results_folder = self.results_folder)
                            except Exception as e:
                                print("Exception in executing callback function\n", e)
                            print(f"Saved model at step {self.step}")

                pbar.update(1)
                torch.cuda.synchronize()

        # Save final model
        if self.master_process:
            np.save(f"{self.results_folder}/losses", losses)
            self.save("latest")
            save_losses_fig(losses, self.results_folder)
            self.ema.ema_model.eval()
            model_to_use = self.ema.ema_model.module if isinstance(self.ema.ema_model, DDP) \
                                else self.ema.ema_model
            try:
                with torch.no_grad(), torch.autocast(device_type=device, dtype=typedict[self.mixed_precision_type]):
                    if self.callback_fn is not None:
                        self.callback_fn(idx = "fin",
                                        b = model_to_use, deconvolver = self.deconvolver,
                                        dataloader = self.dl, validation_data = self.validation_data,
                                        losses = losses, device=self.device,
                                        results_folder = self.results_folder)
            except Exception as e:
                print("Exception in executing callback function\n", e)

            print('training complete')

        return losses



                                    # milestone = self.step // self.save_and_sample_every
                                    # data, obs, latents = next(self.dl)
                                    # data = data.to(self.device)
                                    # obs = obs.to(self.device)
                                    # latents = latents.to(self.device)
                                    # latents = latents if self.deconvolver.use_latents else None
                                    # clean = self.deconvolver.transport(model_to_use, obs, latents)
                                    # try:
                                    #     iutils.save_fig(milestone, data, obs, clean, self.results_folder)
                                    # except Exception as e:
                                    #     print("Exception in saving data")
                                    #     print(e)
