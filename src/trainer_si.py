import numpy as np
from pathlib import Path
from multiprocessing import cpu_count

import torch
from torch.utils.data import DataLoader

from torch.optim import Adam

from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup
from utils import *
import interpolant_utils as iutils

# trainer class
class Trainer:
    def __init__(
            self,
            model, 
            deconvolver,
            dataset,
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
            amp = False,
            mixed_precision_type = 'fp16',
            split_batches = True,
            max_grad_norm = 1.,
            num_workers = 1
    ):
        super().__init__()

        # accelerator
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model
        self.model = model
        self.deconvolver = deconvolver

        # sampling and training hyperparameters
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        self.ds = dataset
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
        self.loss_fn = self.accelerator.prepare(self.deconvolver.loss_fn)
       
       
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
                    data = self.deconvolver.push_fwd(data)
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
                        self.save("latest")
                        self.ema.ema_model.eval()

                        with self.accelerator.autocast():
                            with torch.inference_mode():
                                milestone = self.step // self.save_and_sample_every
                                np.save(f"{self.results_folder}/losses", losses)
                                image = next(self.dl).to(device)
                                corrupted = self.deconvolver.push_fwd(image)
                                clean = self.deconvolver.transport(self.model, corrupted)
                                iutils.save_fig(milestone, image, corrupted, clean, self.results_folder)
                                print(f"Saved model at step {self.step}")
            
                pbar.update(1)

        # Save final model
        self.save("latest")
        self.ema.ema_model.eval()
        with self.accelerator.autocast():
            with torch.inference_mode():
                milestone = self.step // self.save_and_sample_every
                np.save(f"{self.results_folder}/losses", losses)
                image = next(self.dl).to(device)
                corrupted = self.deconvolver.push_fwd(image)
                clean = self.deconvolver.transport(self.model, corrupted)
                iutils.save_fig("fin", image, corrupted, clean, self.results_folder)
                accelerator.print(f"Saved model at step {self.step}")

        accelerator.print('training complete')

        return losses
