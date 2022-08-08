"""PyTorch trainer."""

from abc import ABC, abstractmethod
import os
from typing import Tuple, Union, Dict, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.utils.data import Dataset, DistributedSampler, RandomSampler, SequentialSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dist import is_main_process, get_device


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

    def _make_model_optimizer(self, model: nn.Module, optimizer: str, **kwargs) -> \
            Tuple[Union[DDP, nn.Module], Optimizer]:
        if self._distributed:
            model = DDP(model, device_ids=[get_device()], find_unused_parameters=False)

        optimizers: Dict[str, Optimizer.__class__] = {
            'SGD': SGD,
            'Adam': Adam,
            'AdamW': AdamW
        }
        optimizer = optimizers[optimizer](model.parameters(), **kwargs)
        return model, optimizer

    @staticmethod
    def _collate_fn(batch):
        all_known, all_unknown = [], []
        for known, unknown in batch:
            all_known.append(known)
            all_unknown.append(unknown)
        return torch.stack(all_known), torch.stack(all_unknown)

    @abstractmethod
    def train(self, known: Tensor, unknown: Tensor, epochs: int, batch_size: int, shuffle: bool = True,
              save_freq: int = 100):
        dataset = TensorDataset(known, unknown)
        sampler = DistributedSampler(dataset, shuffle=shuffle) if self._distributed else \
            RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        global_step = 0
        os.makedirs(os.path.join(self._ckpt_dir, self._descr), exist_ok=True)
        for i in range(epochs):
            if is_main_process():
                dataloader = tqdm(dataloader)
                dataloader.set_description(f'Epoch[{i}] {self._descr}')
            for step, (known_batch, unknown_batch) in enumerate(dataloader):
                loss_dict, _ = self.run_step(known_batch, unknown_batch)
                global_step += 1
                self._wrap_step(dataloader, loss_dict, f'Epoch[{i}] {self._descr}', global_step, save_freq)

            self._save_checkpoint(i, 'epoch')

    def _wrap_step(self, dataloader: tqdm, loss_dict: Dict[str: float], prefix: str, global_step: int, save_freq: int):
        if is_main_process():
            loss_descr = [f'{n}={v:.4f}' for n, v in loss_dict.items()]
            dataloader.set_description(f'{prefix}: {",".join(loss_descr)}')
            self._writer.add_scalars('train_loss', loss_dict, global_step)
            if global_step % save_freq == 0:
                self._save_checkpoint(global_step // save_freq, 'step')

    @abstractmethod
    def _save_checkpoint(self, idx: int, by: str):
        raise NotImplementedError()

    @abstractmethod
    def run_step(self, known: Tensor, unknown: Tensor) -> Tuple[Dict[str: float], Optional[Tensor]]:
        raise NotImplementedError()

    @abstractmethod
    def calculate_loss(self, pred: Tensor, target: Tensor):
        raise NotImplementedError()
