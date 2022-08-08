"""PyTorch trainer."""

from abc import ABC, abstractmethod
import os
from typing import Tuple, Union, Dict, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, ConstantLR, _LRScheduler as LRScheduler
from torch.utils.data import Dataset, DistributedSampler, RandomSampler, SequentialSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from .dist import is_main_process, get_device, barrier


class TensorDataset(Dataset):
    def __init__(self, known: Tensor, unknown: Tensor):
        assert len(known) == len(unknown)
        self._known, self._unknown = known, unknown

    def __len__(self):
        return len(self._unknown)

    def __getitem__(self, index):
        return self._known[index], self._unknown[index]


class Trainer(ABC):
    def __init__(self, distributed: bool = False, autocast: bool = False,
                 log_dir: str = 'tflog', ckpt_dir: str = 'checkpoints', descr: str = ''):
        self._distributed, self._autocast, self._descr, self._ckpt_dir = distributed, autocast, descr, ckpt_dir
        self._writer = SummaryWriter(log_dir=os.path.join(log_dir, descr))

    def _make_model_optimizer(self, model: nn.Module, optimizer: str = 'AdamW', scheduler: str = 'StepLR',
                              **kwargs) -> Tuple[Union[DDP, nn.Module], Optimizer, LRScheduler, Optional[GradScaler]]:
        if self._distributed:
            model = DDP(model, device_ids=[get_device()], find_unused_parameters=False)

        optimizers: Dict[str, Optimizer.__class__] = {
            'SGD': SGD,
            'Adam': Adam,
            'AdamW': AdamW
        }
        optimizer = optimizers[optimizer](
            model.parameters(),
            **{n[6:]: v for n, v in kwargs.items() if n.startswith('optim_')}
        )

        schedulers: Dict[str, LRScheduler] = {
            'StepLR': StepLR,
            'ConstantLR': ConstantLR
        }
        lr_scheduler = schedulers[scheduler](
            optimizer,
            **{n[6:]: v for n, v in kwargs.items() if n.startswith('sched_')}
        )

        scaler = None
        if self._distributed:
            scaler = GradScaler(
                enabled=self._autocast,
                **{n[7:]: v for n, v in kwargs.items() if n.startswith('scaler_')}
            )

        return model, optimizer, lr_scheduler, scaler

    @staticmethod
    def _take_step(loss: Tensor, optimizer: Optimizer, grad_scaler: Optional[GradScaler], lr_scheduler: LRScheduler):
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            lr_scheduler.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    @staticmethod
    def _collate_fn(batch):
        all_known, all_unknown = [], []
        for known, unknown in batch:
            all_known.append(known)
            all_unknown.append(unknown)
        return torch.stack(all_known), torch.stack(all_unknown)

    def train(self, known: Tensor, unknown: Tensor, epochs: int, batch_size: int, shuffle: bool = True,
              save_freq: int = 100, resume: bool = True):
        dataset = TensorDataset(known, unknown)
        sampler = DistributedSampler(dataset, shuffle=shuffle) if self._distributed else \
            RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True,
                                collate_fn=self._collate_fn)

        os.makedirs(os.path.join(self._ckpt_dir, self._descr), exist_ok=True)
        epoch, global_step = 0, 0
        if resume:
            epoch, global_step = self._resume_id()
            global_step *= save_freq
        self._reload_checkpoint(epoch, 'epoch')

        for i in range(epoch, epochs, 1):
            if is_main_process():
                dataloader = tqdm(dataloader)
                dataloader.set_description(f'Epoch[{i}] {self._descr}')
            base_step = i * len(dataloader)
            for step, (known_batch, unknown_batch) in enumerate(dataloader):
                if base_step < global_step:
                    continue
                if base_step == global_step:
                    self._reload_checkpoint(global_step // save_freq, 'step')
                loss_dict, _ = self.run_step(known_batch, unknown_batch)
                global_step += 1
                base_step += 1
                self._wrap_step(dataloader, loss_dict, f'Epoch[{i}] {self._descr}', global_step, save_freq)

            barrier()
            if is_main_process():
                self._save_checkpoint(i+1, 'epoch')

    def _resume_id(self) -> Tuple[int, int]:
        all_ckpt = os.listdir(os.path.join(self._ckpt_dir, self._descr))
        epoch_ckpt, batch_ckpt = 0, 0
        for ckpt in all_ckpt:
            if ckpt.startswith('epoch'):
                epoch_id = int(ckpt[6:-3])
                epoch_ckpt = max(epoch_id, epoch_ckpt)
            elif ckpt.startswith('step'):
                step_id = int(ckpt[5:-3])
                batch_ckpt = max(step_id, batch_ckpt)
        return epoch_ckpt, batch_ckpt

    @abstractmethod
    def _reload_checkpoint(self, idx: int, by: str):
        raise NotImplementedError()

    def _wrap_step(self, dataloader: tqdm, loss_dict: Dict[str: float], prefix: str, global_step: int, save_freq: int):
        if is_main_process():
            loss_descr = [f'{n}={v:.4f}' for n, v in loss_dict.items()]
            dataloader.set_description(f'{prefix}: {",".join(loss_descr)}')
            self._writer.add_scalars('train_loss', loss_dict, global_step)
        if global_step % save_freq == 0:
            barrier()
            if is_main_process():
                self._save_checkpoint(global_step // save_freq, 'step')

    @abstractmethod
    def _save_checkpoint(self, idx: int, by: str):
        raise NotImplementedError()

    @abstractmethod
    def run_step(self, known: Tensor, unknown: Tensor) -> Tuple[Dict[str: float], Optional[Tensor]]:
        raise NotImplementedError()
