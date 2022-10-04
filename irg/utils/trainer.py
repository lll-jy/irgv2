"""PyTorch trainer."""

from abc import ABC, abstractmethod
from itertools import chain
import os
from typing import Tuple, Union, Dict, Optional, List, Any

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, ConstantLR, MultiStepLR, ExponentialLR, LinearLR, _LRScheduler as LRScheduler
from torch.utils.data import TensorDataset, DistributedSampler, RandomSampler, SequentialSampler, DataLoader
# from torch.utils.tensorboard import SummaryWriter # TODO
from torch.cuda.amp import GradScaler, GradScaler as SummaryWriter
from tqdm import tqdm

from .dist import is_main_process, get_device, barrier


class InferenceOutput(ABC):
    """
    Inference output structure.
    """
    def __init__(self, output: Tensor):
        self.output = output
        """Output as single Tensor."""


class Trainer(ABC):
    """PyTorch Trainer helper."""
    def __init__(self, distributed: bool = False, autocast: bool = False,
                 log_dir: str = 'tflog', ckpt_dir: str = 'checkpoints', descr: str = ''):
        """
        **Args**:

        - `distributed` (`bool`): Whether distributed training is used. Default is `False`.
        - `autocast` (`bool`): Whether to autocast. Default is `bool`.
        - `log_dir` (`str`): The directory saving tensorboard.
        - `ckpt_dir` (`str`): The directory saving checkpoints.
        - `descr` (`str`): The description of this training.
          Tensorboard files are saved under log_dir/descr/, and checkpoints are saved under ckpt_dir/descr/.
        """
        self._distributed, self._autocast, self._descr, self._ckpt_dir = distributed, autocast, descr, ckpt_dir
        self._device = get_device()
        self._writer = SummaryWriter(log_dir=os.path.join(log_dir, descr))

    def _make_model_optimizer(self, model: Union[nn.Module, List[nn.Module]], optimizer: str = 'AdamW',
                              scheduler: str = 'StepLR', **kwargs) -> Tuple[
            Union[Union[DDP, nn.Module], List[Union[DDP, nn.Module]]], Optimizer, LRScheduler, Optional[GradScaler]]:
        is_single = not isinstance(model, List)
        if self._distributed:
            if is_single:
                model = DDP(model, device_ids=[self._device], find_unused_parameters=False)
            else:
                model = [DDP(m, device_ids=[self._device], find_unused_parameters=False) for m in model]

        optimizers: Dict[str, Optimizer.__class__] = {
            'SGD': SGD,
            'Adam': Adam,
            'AdamW': AdamW
        }
        optimizer = optimizers[optimizer](
            model.parameters() if is_single else chain(*[m.parameters() for m in model]),
            **{n[6:]: v for n, v in kwargs.items() if n.startswith('optim_')}
        )

        schedulers: Dict[str, LRScheduler] = {
            'StepLR': StepLR,
            'ConstantLR': ConstantLR,
            'MultiStepLR': MultiStepLR,
            'ExponentialLR': ExponentialLR,
            'LinearLR': LinearLR
        }
        default_scheduler_args: Dict[str, Dict[str, Any]] = {
            'StepLR': {'step_size': 100},
            'ConstantLR': {},
            'MultiStepLR': {'milestones': [100, 300]},
            'ExponentialLR': {'gamma': 0.99},
            'LinearLR': {}
        }
        lr_scheduler = schedulers[scheduler](
            optimizer,
            **(default_scheduler_args[scheduler] | {n[6:]: v for n, v in kwargs.items() if n.startswith('sched_')})
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

    def _make_dataloader(self, known: Tensor, unknown: Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
        dataset = TensorDataset(known, unknown)
        sampler = DistributedSampler(dataset, shuffle=shuffle) if self._distributed else \
            RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True,
                                collate_fn=self._collate_fn)
        return dataloader

    def train(self, known: Tensor, unknown: Tensor, epochs: int = 10, batch_size: int = 100, shuffle: bool = True,
              save_freq: int = 100, resume: bool = True):
        """
        Train the model given data.

        **Args**:

        - `known` (`torch.Tensor`): Known part of data as tensor.
        - `unknown` (`torch.Tensor`): Unknown part of data as tensor.
        - `epochs` (`int`): Number of epochs to train. Default is 100.
        - `batch_size` (`int`): Batch size per GPU/CPU. Default is 100.
        - `shuffle` (`bool`): Whether to shuffle the data. Default is True.
        - `save_freq` (`int`): Save checkpoint frequency (every how many steps). Default is 100.
        - `resume` (`bool`): Whether to resume from trained result (from ckpt_dir).
        """
        dataloader = self._make_dataloader(known, unknown, batch_size, shuffle)

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

    def _wrap_step(self, dataloader: tqdm, loss_dict: Dict[str, float], prefix: str, global_step: int, save_freq: int):
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
    def run_step(self, known: Tensor, unknown: Tensor) -> Tuple[Dict[str, float], Optional[Tensor]]:
        """
        Run one step for one batch.

        **Args**:

        - `known` (`torch.Tensor`): Known part of the batch as tensor.
        - `unknown` (`torch.Tensor`): Unknown part of the batch as tensor.

        **Return**: (Dict of loss name to loss float number, inference result tensor).
        """
        raise NotImplementedError()

    @abstractmethod
    def inference(self, known: Tensor, batch_size: int) -> InferenceOutput:
        """
        Infer using the trained model.

        **Args**:

        - `known` (`torch.Tensor`): The input to the model.
        - `batch_size` (`int`): Batch size for inference.

        **Return**: Inference result.
        """
        raise NotImplementedError()
