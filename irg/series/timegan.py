"""Series generation using TimeGAN trainer."""
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import SeriesTrainer
from ..utils.torch import TimeGanNet
from ..utils import dist

_LOGGER = logging.getLogger()


class TimeGANTrainer(SeriesTrainer):
    """Trainer for series table generation for TimeGAN."""
    def __init__(self, hidden_dim: int = 40, n_layers: int = 1, **kwargs):
        super().__init__(**kwargs)
        self._embedder = TimeGanNet(
            input_size=self._known_dim + self._unknown_dim + 1, output_size=hidden_dim,
            hidden_dim=hidden_dim, n_layers=n_layers, activation='sigmoid'
        ).to(self._device)
        self._recovery = TimeGanNet(
            input_size=hidden_dim, output_size=self._unknown_dim + 1, hidden_dim=hidden_dim, n_layers=n_layers
        ).to(self._device)
        self._generator = TimeGanNet(
            input_size=self._known_dim + self._unknown_dim + 1, output_size=hidden_dim,
            hidden_dim=hidden_dim, n_layers=n_layers, activation='sigmoid'
        ).to(self._device)
        self._supervisor = TimeGanNet(
            input_size=hidden_dim, output_size=hidden_dim, hidden_dim=hidden_dim, n_layers=n_layers,
            activation='sigmoid'
        ).to(self._device)
        self._discriminator = TimeGanNet(
            input_size=hidden_dim, output_size=1, hidden_dim=hidden_dim, n_layers=n_layers,
        ).to(self._device)

        self._embedder, self._opt_e, self._lrs_e, self._gs_e = self._make_model_optimizer(
            self._embedder, **kwargs
        )
        self._recovery, self._opt_r, self._lrs_r, self._gs_r = self._make_model_optimizer(
            self._recovery, **kwargs
        )
        self._generator, self._opt_g, self._lrs_g, self._gs_g = self._make_model_optimizer(
            self._generator, **kwargs
        )
        self._supervisor, self._opt_s, self._lrs_s, self._gs_s = self._make_model_optimizer(
            self._supervisor, **kwargs
        )
        self._discriminator, self._opt_d, self._lrs_d, self._gs_d = self._make_model_optimizer(
            self._discriminator, **kwargs
        )
        self._hidden_dim = hidden_dim
        self._pretrained = False, False

    def _load_content_from(self, loaded: Dict[str, Any]):
        super()._load_content_from(loaded)
        self._embedder, self._opt_e, self._lrs_e, self._gs_e = self._load_state_dict(
            self._embedder, self._opt_e, self._lrs_e, self._gs_e, loaded['embedder']
        )
        self._recovery, self._opt_r, self._lrs_r, self._gs_r = self._load_state_dict(
            self._recovery, self._opt_r, self._lrs_r, self._gs_r, loaded['recovery']
        )
        self._generator, self._opt_g, self._lrs_g, self._gs_g = self._load_state_dict(
            self._generator, self._opt_g, self._lrs_g, self._gs_g, loaded['generator']
        )
        self._supervisor, self._opt_s, self._lrs_s, self._gs_s = self._load_state_dict(
            self._supervisor, self._opt_s, self._lrs_s, self._gs_s, loaded['supervisor']
        )
        self._discriminator, self._opt_d, self._lrs_d, self._gs_d = self._load_state_dict(
            self._discriminator, self._opt_d, self._lrs_d, self._gs_d, loaded['discriminator']
        )
        self._pretrained = loaded['pretrained']

    def _construct_content_to_save(self) -> Dict[str, Any]:
        return {
            'embedder': self._full_state_dict(self._embedder, self._opt_e, self._lrs_e, self._gs_e),
            'recovery': self._full_state_dict(self._recovery, self._opt_r, self._lrs_r, self._gs_r),
            'generator': self._full_state_dict(self._generator, self._opt_g, self._lrs_g, self._gs_g),
            'supervisor': self._full_state_dict(self._supervisor, self._opt_s, self._lrs_s, self._gs_s),
            'discriminator': self._full_state_dict(self._discriminator, self._opt_d, self._lrs_d, self._gs_d),
            'pretrained': self._pretrained
        } | super()._construct_content_to_save()

    @classmethod
    def _reconstruct(cls, distributed: bool, autocast: bool, log_dir: str, ckpt_dir: str, descr: str,
                     known_dim: int, unknown_dim: int, cat_dims: List[Tuple[int, int]], lae_trained: bool,
                     lae: nn.Module, optimizer_lae: Optimizer, lr_schd_lae: LRScheduler,
                     grad_scaler_lae: Optional[GradScaler], corr_mat: Optional[Tensor],
                     corr_mask: Optional[Tensor], mean: Optional[Tensor],
                     base_ids: Tuple[List[int], List[int]], seq_ids: Tuple[List[int], List[int]],
                     embedder: TimeGanNet, opt_e: Optimizer, lrs_e: LRScheduler, gs_e: Optional[GradScaler],
                     recovery: TimeGanNet, opt_r: Optimizer, lrs_r: LRScheduler, gs_r: Optional[GradScaler],
                     generator: TimeGanNet, opt_g: Optimizer, lrs_g: LRScheduler, gs_g: Optional[GradScaler],
                     supervisor: TimeGanNet, opt_s: Optimizer, lrs_s: LRScheduler, gs_s: Optional[GradScaler],
                     discriminator: TimeGanNet, opt_d: Optimizer, lrs_d: LRScheduler, gs_d: Optional[GradScaler],
                     hidden_dim: int, pretrained: (bool, bool)) ->\
            "TimeGANTrainer":
        base = SeriesTrainer._reconstruct(
            distributed, autocast, log_dir, ckpt_dir, descr,
            known_dim, unknown_dim, cat_dims, lae_trained,
            lae, optimizer_lae, lr_schd_lae, grad_scaler_lae,
            corr_mat, corr_mask, mean, base_ids, seq_ids
        )
        base.__class__ = TimeGANTrainer
        base._embedder, base._opt_e, base._lrs_e, base._gs_e = embedder, opt_e, lrs_e, gs_e
        base._recovery, base._opt_r, base._lrs_r, base._gs_r = recovery, opt_r, lrs_r, gs_r
        base._generator, base._opt_g, base._lrs_g, base._gs_g = generator, opt_g, lrs_g, gs_g
        base._supervisor, base._opt_s, base._lrs_s, base._gs_s = supervisor, opt_s, lrs_s, gs_s
        base._discriminator, base._opt_d, base._lrs_d, base._gs_d = discriminator, opt_d, lrs_d, gs_d
        base._hidden_dim, base._pretrained = hidden_dim, pretrained
        return base

    def __reduce__(self):
        _, var = super().__reduce__()
        return self._reconstruct, var + (
            self._embedder, self._opt_e, self._lrs_e, self._gs_e,
            self._recovery, self._opt_r, self._lrs_r, self._gs_r,
            self._generator, self._opt_g, self._lrs_g, self._gs_g,
            self._supervisor, self._opt_s, self._lrs_s, self._gs_s,
            self._discriminator, self._opt_d, self._lrs_d, self._gs_d,
            self._hidden_dim, self._pretrained
        )

    def train(self, known: Tensor, unknown: Tensor, epochs: int = 10, batch_size: int = 100, shuffle: bool = True,
              save_freq: int = 100, resume: bool = True, lae_epochs: int = 10):
        (epoch, global_step), dataloader = self._prepare_training(known, unknown, batch_size, shuffle, resume)
        self._run_embedding(dataloader, epochs)
        self._run_training(epoch, epochs, global_step, save_freq, dataloader)

    def _run_embedding(self, dataloader: DataLoader, epochs: int = 10):
        if self._pretrained[0]:
            return
        self._embedder.train()
        self._recovery.train()
        for i in range(epochs):
            descr = f'Epoch[{i}] {self._descr} Emb'
            if dist.is_main_process():
                dataloader = tqdm(dataloader)
                dataloader.set_description(descr)
            for step, (known_batch, unknown_batch) in enumerate(dataloader):
                data = torch.cat([known_batch[:, :, :-1], unknown_batch], dim=-1)
                h, _ = self._embedder(data)
                h = h.view(*data.shape[:2], self._hidden_dim)
                x_tilde, _ = self._recovery(h)
                x_tilde = x_tilde.view(*data.shape[:2], -1)

                e_loss = self._calculate_recon_loss(data[:, :, known_batch.shape[-1]-1:], x_tilde, True)
                self._take_step(e_loss, self._opt_e, self._gs_e, self._lrs_e, retain_graph=True)
                self._take_step(e_loss, self._opt_r, self._gs_r, self._lrs_r, retain_graph=True)
                if dist.is_main_process():
                    dataloader.set_description(f'{descr} Emb: recon_loss: {e_loss.item():.4f}')
            dist.barrier()
            if dist.is_main_process():
                self._save_checkpoint(0, 'final', 0, 0)

        if dist.is_main_process():
            self._save_checkpoint(0, 'final', 0, 0)
        self._pretrained = True, False
        _LOGGER.debug('Finished pretraining embedding.')

    def _run_supervised(self, dataloader: DataLoader, epochs: int = 10):
        if self._pretrained[1]:
            return
        self._embedder.train()
        self._supervisor.train()
        for i in range(epochs):
            descr = f'Epoch[{i}] {self._descr} Sup'
            if dist.is_main_process():
                dataloader = tqdm(dataloader)
                dataloader.set_description(descr)
            for step, (known_batch, unknown_batch) in enumerate(dataloader):
                data = torch.cat([known_batch[:, :, :-1], unknown_batch], dim=-1)
                h, _ = self._embedder(data)
                h = h.view(*data.shape[:2], self._hidden_dim)
                h_hat, _ = self._supervisor(h)
                h_hat = h_hat.view(*data.shape[:2], -1)

                g_loss_s = F.mse_loss(h[:, 1:, :], h_hat[:, :-1, :])
                self._take_step(g_loss_s, self._opt_e, self._gs_e, self._lrs_e, retain_graph=True)
                self._take_step(g_loss_s, self._opt_s, self._gs_s, self._lrs_s, retain_graph=True)
                if dist.is_main_process():
                    dataloader.set_description(f'{descr} Sup: sup_loss: {g_loss_s.item():.4f}')
            dist.barrier()
            if dist.is_main_process():
                self._save_checkpoint(0, 'final', 0, 0)

        if dist.is_main_process():
            self._save_checkpoint(0, 'final', 0, 0)
        self._pretrained = True
        _LOGGER.debug('Finished pretraining supervised')

    def run_step(self, batch: Tuple[Tensor, ...]) -> Tuple[Dict[str, float], Optional[Tensor]]:
        pass
