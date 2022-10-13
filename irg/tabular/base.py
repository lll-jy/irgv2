"""(Partial) tabular generator training."""

from abc import ABC
from typing import Tuple, List, Optional

import torch
from torch import Tensor

from ..utils import Trainer


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
        - `kwargs`: Inherited arguments from [`Trainer`](../utils#irg.utils.Trainer).
        """
        super().__init__(**kwargs)
        self._known_dim, self._unknown_dim = known_dim, unknown_dim
        self._cat_dims = sorted(cat_dims)
        self._fitted_mean: Optional[Tensor] = None
        self._fitted_std: Optional[Tensor] = None
        self._total_train: int = 0
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

    def train(self, known: Tensor, unknown: Tensor, epochs: int = 10, batch_size: int = 100, shuffle: bool = True,
              save_freq: int = 100, resume: bool = True):
        self._fit_mean_std(unknown)
        super().train(known, unknown, epochs, batch_size, shuffle, save_freq, resume)

    def _fit_mean_std(self, x: Tensor):
        x = x.to(self._device)
        self._fitted_mean = x.mean(dim=0)
        self._fitted_std = x.std(dim=0)
        self._total_train = x.shape[0]

    @classmethod
    def _reconstruct(cls, distributed: bool, autocast: bool, log_dir: str, ckpt_dir: str, descr: str,
                     cat_dims: List[Tuple[int, int]], known_dim: int, unknown_dim: int,
                     fitted_mean: Tensor, fitted_std: Tensor, total_train: int) -> "TabularTrainer":
        base = super()._reconstruct(distributed, autocast, log_dir, ckpt_dir, descr)
        base.__class__ = TabularTrainer
        base._known_dim, base._unknown_dim, base._cat_dims = known_dim, unknown_dim, cat_dims
        base._fitted_mean, base._fitted_std, base._total_train = fitted_mean, fitted_std, total_train
        return base

    def __reduce__(self):
        _, var = super().__reduce__()
        return self._reconstruct, var + (self._known_dim, self._unknown_dim, self._cat_dims,
                                         self._fitted_mean, self._fitted_std, self._total_train)

    def _make_noise(self, x: Tensor):
        if x.shape[0] == 1:
            return x
        x_mean, x_std = x.mean(dim=0), x.std(dim=0)
        noise_mean = (self._fitted_mean - x_mean).repeat(x.shape[0]).reshape(-1, self._unknown_dim)
        noise_std = torch.sqrt((self._fitted_std ** 2
                                / self._total_train * (self._total_train - 1) * x.shape[0] / (x.shape[0] - 1)
                                - x_std ** 2).abs()).repeat(x.shape[0]).reshape(-1, x.shape[1])
        noise = torch.normal(noise_mean, noise_std)
        return noise
