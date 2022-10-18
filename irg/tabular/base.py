"""(Partial) tabular generator training."""

from abc import ABC
from typing import Tuple, List, Optional, Dict, Any

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import Trainer
from ..utils.torch import LinearAutoEncoder
from ..utils import dist


class TabularTrainer(Trainer, ABC):
    """Trainer for tabular models."""
    def __init__(self, cat_dims: List[Tuple[int, int]], known_dim: int, unknown_dim: int, **kwargs):
        """
        **Args**:

        - `cat_dims` (`List[Tuple[int, int]]`): Dimensions corresponding to one categorical column.
          For example, the table has 1 categorical column with 3 categories, and 1 numerical column with 2 clusters, in
          this order. The normalized columns is something like [col_1_is_nan, col_1_cat_1, col_1_cat_2, col_1_cat_3,
          col_2_is_nan, col_2_value, col_2_cluster_1, col_2_cluster_2]. Among them, is_nan columns are categorical
          columns on their own, which will be applied sigmoid as activate function. Cluster and category columns are
          categorical column groups (at least 2 columns), which will be applied Gumbel softmax as activate functions.
          The value column is not categorical, so it will be applied tanh as activate function. The ranges are described
          in left-closed-right-open manner. In this example, the input should be [(0, 1), (1, 4), (4, 5), (6, 8)].
        - `known_dim` (`int`): Number of dimensions in total for known columns.
        - `unknown_dim` (`int`): Number of dimensions in total for unknown columns.
        - `kwargs`: It has the following groups:
            - Inherited arguments from [`Trainer`](../utils#irg.utils.Trainer).
            - AutoEncoder arguments, all prefixed with "lae_" (for example, argument "arg1" under this group will be
              named as "lae_arg1").
                - `optimizer` (`str`): Optimizer type, currently support "SGD", "Adam", and "AdamW" only.
                  Default is "AdamW".
                - `scheduler` (`str`): LR scheduler type, currently support "StepLR" and "ConstantLR" only.
                  Default is "StepLR".
                - Optimizer constructor arguments, all prefixed with "optim_". (That is, argument "arg1" under this
                  group will be named as "gen_optim_arg1".
                - Scheduler constructor arguments, all prefixed with "sched_".
                - GradScaler constructor arguments, all prefixed with "scaler_".
        """
        super().__init__(**kwargs)
        self._known_dim, self._unknown_dim = known_dim, unknown_dim
        self._cat_dims = sorted(cat_dims)
        if self._known_dim > 0:
            self._lae = LinearAutoEncoder(
                context_dim=self._known_dim,
                full_dim=self._unknown_dim
            ).to(self._device)
            self._lae, self._optimizer_lae, self._lr_schd_lae, self._grad_scaler_lae = self._make_model_optimizer(
                self._lae,
                **{n[4:]: v for n, v in kwargs.items() if n.startswith('lae_')}
            )
        else:
            self._lae, self._optimizer_lae, self._lr_schd_lae, self._grad_scaler_lae = None, None, None, None
        self._lae_trained = False
        self._corr_mat, self._corr_mask = None, None
        self._mean = None
        if not self._validate_cat_dims(self._cat_dims):
            raise ValueError('Category dimensions should be disjoint.')

    @staticmethod
    def _validate_cat_dims(cat_dims) -> bool:
        pre = 0
        for l, r in cat_dims:
            if l < pre or r <= l:
                return False
            pre = r
        return True

    @property
    def unknown_dim(self) -> int:
        """Number of unknown dimensions"""
        return self._unknown_dim

    def _calculate_corr(self, known: Tensor, unknown: Tensor):
        self._corr_mat = torch.corrcoef(torch.cat([known, unknown], dim=1).permute(1, 0))[-self.unknown_dim:]
        nan_mask = ~self._corr_mat.isnan()
        feature_mask = ~torch.zeros_like(nan_mask, dtype=torch.bool, device=self._device)
        for l, r in self._cat_dims:
            feature_mask[l:r, l-self._unknown_dim:r-self._unknown_dim] = 0
        value_mask = self._corr_mat ** 2 > 0.8
        self._corr_mask = nan_mask & feature_mask & value_mask
        self._mean = unknown.mean(dim=0)

    def train(self, known: Tensor, unknown: Tensor, epochs: int = 10, batch_size: int = 100, shuffle: bool = True,
              save_freq: int = 100, resume: bool = True, lae_epochs: int = 10):
        self._calculate_corr(known, unknown)
        (epoch, global_step), dataloader = self._prepare_training(known, unknown, batch_size, shuffle, resume)
        if not self._lae_trained and self._known_dim != 0:
            self._train_lae(dataloader, lae_epochs)
        self._run_training(epoch, epochs, global_step, save_freq, dataloader)

    def _train_lae(self, dataloader: DataLoader, epochs: int = 10):
        self._lae.train()
        for i in range(epochs):
            descr = f'Epoch[{i}] {self._descr} LAE'
            if dist.is_main_process():
                dataloader = tqdm(dataloader)
                dataloader.set_description(descr)
            for step, (known_batch, unknown_batch) in enumerate(dataloader):
                known, unknown = known_batch.to(self._device), unknown_batch.to(self._device)
                recon = self._lae(known, 'recon')
                real = torch.cat([known, unknown], dim=1)
                loss = F.mse_loss(recon, real)
                self._take_step(loss, self._optimizer_lae, self._grad_scaler_lae, self._lr_schd_lae)
                if dist.is_main_process():
                    dataloader.set_description(f'{descr}: recon_loss: {loss.item():.4f}')

            dist.barrier()
            if dist.is_main_process():
                self._save_checkpoint(0, 'final', 0, 0)

        if dist.is_main_process():
            self._save_checkpoint(0, 'final', 0, 0)
        self._lae.eval()
        self._lae_trained = True

    def _make_context(self, known: Tensor):
        if self._known_dim == 0:
            return known
        with torch.no_grad():
            encoded = self._lae(known, 'enc')
        return encoded

    def _load_content_from(self, loaded: Dict[str, Any]):
        if self._known_dim > 0:
            self._lae, self._optimizer_lae, self._lr_schd_lae, self._grad_scaler_lae = self._load_state_dict(
                self._lae, self._optimizer_lae, self._lr_schd_lae, self._grad_scaler_lae, loaded['lae']
            )
        self._lae_trained = loaded['lae_trained']

    def _construct_content_to_save(self) -> Dict[str, Any]:
        return {
            'lae': self._full_state_dict(
                self._lae, self._optimizer_lae, self._lr_schd_lae, self._grad_scaler_lae
            ) if self._known_dim > 0 else None,
            'seed': torch.initial_seed(),
            'lae_trained': self._lae_trained
        }

    @classmethod
    def _reconstruct(cls, distributed: bool, autocast: bool, log_dir: str, ckpt_dir: str, descr: str,
                     known_dim: int, unknown_dim: int, cat_dims: List[Tuple[int, int]], lae_trained: bool,
                     lae: nn.Module, optimizer_lae: Optimizer, lr_schd_lae: LRScheduler,
                     grad_scaler_lae: Optional[GradScaler], corr_mat: Optional[Tensor],
                     corr_mask: Optional[Tensor], mean: Optional[Tensor], **kwargs) -> "TabularTrainer":
        base = super()._reconstruct(distributed, autocast, log_dir, ckpt_dir, descr)
        base.__class__ = TabularTrainer
        base._known_dim, base._unknown_dim, base._cat_dims = known_dim, unknown_dim, cat_dims
        base._lae_trained = lae_trained
        base._lae, base._optimizer_lae, base._lr_schd_lae, base._grad_scaler_lae = (
            lae, optimizer_lae, lr_schd_lae, grad_scaler_lae)
        base._corr_mat, base._corr_mask, base._mean = corr_mat, corr_mask, mean
        return base

    def __reduce__(self):
        _, var = super().__reduce__()
        return self._reconstruct, var + (
            self._known_dim, self._unknown_dim, self._cat_dims, self._lae_trained,
            self._lae, self._optimizer_lae, self._lr_schd_lae, self._grad_scaler_lae,
            self._corr_mat, self._corr_mask, self._mean
        )

    @property
    def _all_cat_cols(self) -> List[int]:
        res = []
        for l, r in self._cat_dims:
            res += [*range(l, r)]
        return res

    @staticmethod
    def _check_nan(tensor: Tensor, descr: str):
        if tensor.isnan().sum() > 0:
            print(descr, 'has nan')
            raise ValueError()

    def _meta_loss(self, known: Tensor, real: Tensor, fake: Tensor) -> Tensor:
        fake_mean = fake.mean(dim=0)
        real_mean = (real.mean(dim=0) + self._mean) / 2
        mean_diff = real_mean - fake_mean
        mean_loss = (mean_diff ** 2).sum()

        full_fake = torch.cat([known, fake], dim=1)
        # full_real, full_fake = torch.cat([known, real], dim=1), torch.cat([known, fake], dim=1)

        # real_corr = torch.corrcoef(full_real.permute(1, 0))[-self.unknown_dim:]
        fake_corr = torch.corrcoef(full_fake.permute(1, 0))[-self.unknown_dim:]
        # nan_mask = ~real_corr.isnan()
        # feature_mask = ~torch.zeros_like(nan_mask, dtype=torch.bool, device=self._device)
        # for l, r in self._cat_dims:
        #     feature_mask[l:r, l-self._unknown_dim:r-self._unknown_dim] = 0
        # value_mask = real_corr ** 2 > 0.8
        # mask = nan_mask & feature_mask & value_mask
        # corr_loss = ((real_corr[mask] - fake_corr[mask]) ** 2).sum()
        mask = self._corr_mask & (~fake_corr.isnan())
        corr_loss = ((self._corr_mat[mask] - fake_corr[mask]) ** 2).sum()
        self._check_nan(mean_loss, 'mean loss')
        self._check_nan(corr_loss, 'cor loss')
        return mean_loss + corr_loss
