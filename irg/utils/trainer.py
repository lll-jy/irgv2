"""PyTorch trainer."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from itertools import chain
import logging
import os
from typing import Tuple, Union, Dict, Optional, List, Any

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, ConstantLR, MultiStepLR, ExponentialLR, LinearLR, _LRScheduler as LRScheduler
from torch.utils.data import TensorDataset, DistributedSampler, RandomSampler, SequentialSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from .dist import is_main_process, get_device, barrier, to_device

_LOGGER = logging.getLogger()


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
        self._distributed, self._autocast, self._descr = distributed, autocast, descr
        self._ckpt_dir, self._log_dir = ckpt_dir, log_dir
        self._device = get_device()
        self._writer = SummaryWriter(log_dir=os.path.join(log_dir, descr))
        os.makedirs(os.path.join(self._ckpt_dir, self._descr), exist_ok=True)

    @classmethod
    def _reconstruct(cls, distributed: bool, autocast: bool, log_dir: str, ckpt_dir: str, descr: str, **kwargs) ->\
            "Trainer":
        base = _DummyEmptyTrainer()
        base.__class__ = Trainer
        base._distributed, base._autocast, base._descr = distributed, autocast, descr
        base._ckpt_dir, base._log_dir = ckpt_dir, log_dir
        base._device = get_device()
        base._writer = None
        return base

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
    def _load_state_dict(model: nn.Module, optimizer: Optimizer, lr_schd: LRScheduler,
                         grad_scaler: Optional[GradScaler], loaded: Dict[str, Any]):
        model_dict = loaded['model']
        if hasattr(model, 'module'):
            model_dict = OrderedDict({f'module.{n}': v for n, v in model_dict.items()})
        model.load_state_dict(model_dict, strict=True)
        optimizer.load_state_dict(loaded['optimizer'])
        lr_schd.load_state_dict(loaded['lr_scheduler'])
        if 'grad_scaler' in loaded:
            grad_scaler.load_state_dict(loaded['grad_scaler'])
        return model, optimizer, lr_schd, grad_scaler

    @staticmethod
    def _full_state_dict(model: nn.Module, optimizer: Optimizer, lr_schd: LRScheduler,
                         grad_scaler: Optional[GradScaler]) -> Dict[str, Any]:
        return {
            'model': (model.module if hasattr(model, 'module') else model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_schd.state_dict()
        } | ({'grad_scaler': grad_scaler.state_dict()} if grad_scaler is not None else {})

    @staticmethod
    def _take_step(loss: Tensor, optimizer: Optimizer, grad_scaler: Optional[GradScaler], lr_scheduler: LRScheduler,
                   retain_graph: bool = False, do_zero_grad: bool = True):
        if do_zero_grad:
            optimizer.zero_grad()
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward(retain_graph=retain_graph)
            grad_scaler.unscale_(optimizer)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            lr_scheduler.step()
        else:
            loss.backward(retain_graph=retain_graph)
            optimizer.step()
            lr_scheduler.step()

    def __reduce__(self):
        return self.__class__, (self._distributed, self._autocast, self._log_dir, self._ckpt_dir, self._descr)

    def _collate_fn(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        all_known, all_unknown = [], []
        for known, unknown in batch:
            all_known.append(known)
            all_unknown.append(unknown)
        return torch.stack(all_known), torch.stack(all_unknown)

    def _collate_fn_infer(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        all_known = []
        for known, in batch:
            all_known.append(known)
        return torch.stack(all_known),

    def _make_dataloader(self, known: Tensor, unknown: Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
        dataset = TensorDataset(known, unknown)
        sampler = DistributedSampler(dataset, shuffle=shuffle) if self._distributed else \
            RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True,
                                collate_fn=self._collate_fn)
        return dataloader

    def _make_infer_dataloader(self, known: Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
        dataset = TensorDataset(known)
        sampler = DistributedSampler(dataset, shuffle=shuffle) if self._distributed else \
            RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True,
                                collate_fn=self._collate_fn_infer)
        return dataloader

    def _prepare_training(self, known: Tensor, unknown: Tensor, batch_size: int = 100, shuffle: bool = True,
                          resume: bool = True) -> ((int, int), DataLoader):
        dataloader = self._make_dataloader(known, unknown, batch_size, shuffle)
        os.makedirs(os.path.join(self._ckpt_dir, self._descr), exist_ok=True)
        if resume:
            steps, epochs = self._reload_checkpoint(0, 'final')
        else:
            steps, epochs = 0, 0
        return (steps, epochs), dataloader

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
        (epoch, global_step), dataloader = self._prepare_training(known, unknown, batch_size, shuffle, resume)
        self._run_training(epoch, epochs, global_step, save_freq, dataloader)

    def _run_training(self, epoch: int, epochs: int, global_step: int, save_freq: int, dataloader: DataLoader):
        for i in range(epoch, epochs, 1):
            if is_main_process():
                dataloader = tqdm(dataloader)
                dataloader.set_description(f'Epoch[{i}] {self._descr}')
            base_step = i * len(dataloader)
            for step, batch in enumerate(dataloader):
                if base_step < global_step:
                    continue
                if base_step == global_step:
                    self._reload_checkpoint(global_step // save_freq, 'step')
                batch = tuple(to_device(b, self._device) for b in batch)
                loss_dict, _ = self.run_step(batch)
                global_step += 1
                base_step += 1
                self._wrap_step(dataloader, loss_dict, f'Epoch[{i}] {self._descr}', global_step, save_freq, i + 1)

            barrier()
            if is_main_process():
                self._save_checkpoint(i+1, 'epoch', global_step, i + 1)
                self._save_checkpoint(0, 'final', global_step, i + 1)

        if is_main_process():
            self._save_checkpoint(0, 'final', global_step, epochs)

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

    def _reload_checkpoint(self, idx: int, by: str) -> Tuple[int, int]:
        path = os.path.join(self._ckpt_dir, self._descr, f'{by}_{idx:07d}.pt')
        if not os.path.exists(path):
            return 0, 0
        loaded = torch.load(path)
        self._load_content_from(loaded)
        torch.manual_seed(loaded['seed'])
        steps, epochs = loaded['steps'], loaded['epochs']
        _LOGGER.info(f'Resume at step {steps}, epoch {epochs} from {path}.')
        print(f'Resume at step {steps}, epoch {epochs} from {path}.')
        return steps, epochs

    @abstractmethod
    def _load_content_from(self, loaded: Dict[str, Any]):
        raise NotImplementedError()

    def _wrap_step(self, dataloader: tqdm, loss_dict: Dict[str, float], prefix: str, global_step: int, save_freq: int,
                   epoch: int):
        if is_main_process():
            loss_descr = [f'{n}={v:.4f}' for n, v in loss_dict.items()]
            dataloader.set_description(f'{prefix}: {",".join(loss_descr)}')
            self._writer.add_scalars('train_loss', loss_dict, global_step)
        if global_step % save_freq == 0:
            barrier()
            if is_main_process():
                self._save_checkpoint(global_step // save_freq, 'step', global_step, epoch)

    def _save_checkpoint(self, idx: int, by: str, global_step: int, epoch: int):
        torch.save(
            self._construct_content_to_save() | {'steps': global_step, 'epochs': epoch},
            os.path.join(self._ckpt_dir, self._descr, f'{by}_{idx:07d}.pt')
        )

    @abstractmethod
    def _construct_content_to_save(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def run_step(self, batch: Tuple[Tensor, ...]) -> Tuple[Dict[str, float], Optional[Tensor]]:
        """
        Run one step for one batch.

        **Args**:

        - `batch` (`Tuple[Tensor, ...]`): Content of a batch. Typically contains at least a known and unknown
          part.

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


class _DummyEmptyTrainer(Trainer):
    def __init__(self):
        pass

    def run_step(self, batch: Tuple[Tensor, ...]) -> Tuple[Dict[str, float], Optional[Tensor]]:
        pass

    def _load_content_from(self, loaded: Dict[str, Any]):
        pass

    def _construct_content_to_save(self) -> Dict[str, Any]:
        pass

    def inference(self, known: Tensor, batch_size: int) -> InferenceOutput:
        pass
