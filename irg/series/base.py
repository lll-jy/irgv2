"""Series generator training."""

from abc import ABC
from typing import Tuple, List, Optional

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.cuda.amp import GradScaler

from ..tabular import TabularTrainer


class SeriesTrainer(TabularTrainer, ABC):
    """Trainer for series data models."""
    def __init__(self, base_ids: Tuple[List[int], List[int]], seq_ids: Tuple[List[int], List[int]], **kwargs):
        self._base_ids = base_ids
        self._seq_ids = seq_ids
        super().__init__(**kwargs)

    @classmethod
    def _reconstruct(cls, distributed: bool, autocast: bool, log_dir: str, ckpt_dir: str, descr: str,
                     known_dim: int, unknown_dim: int, cat_dims: List[Tuple[int, int]], lae_trained: bool,
                     lae: nn.Module, optimizer_lae: Optimizer, lr_schd_lae: LRScheduler,
                     grad_scaler_lae: Optional[GradScaler], corr_mat: Optional[Tensor],
                     corr_mask: Optional[Tensor], mean: Optional[Tensor],
                     base_ids: Tuple[List[int], List[int]], seq_ids: Tuple[List[int], List[int]], **kwargs) ->\
            "SeriesTrainer":
        base = super()._reconstruct(
            distributed, autocast, log_dir, ckpt_dir, descr,
            known_dim, unknown_dim, cat_dims, lae_trained,
            lae, optimizer_lae, lr_schd_lae, grad_scaler_lae,
            corr_mat, corr_mask, mean
        )
        base.__class__ = cls
        base._base_ids, base._seq_ids = base_ids, seq_ids
        return base

    def __reduce__(self):
        _, var = super().__reduce__()
        return self._reconstruct, var + (
            self._base_ids, self._seq_ids
        )
