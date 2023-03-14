"""Series generator training."""

from abc import ABC
from typing import Tuple, List, Optional

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from ..tabular import TabularTrainer


class _TensorListDataset(Dataset):
    def __init__(self, *data: List[Tensor]):
        self._data = data
        if len(self._data) > 0:
            self._length = len(self._data[0])
        else:
            self._length = 0

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, item: int) -> Tuple:
        return tuple([d[item] for d in self._data])


class SeriesTrainer(TabularTrainer, ABC):
    """Trainer for series data models."""
    def __init__(self, base_ids: Tuple[List[int], List[int]], seq_ids: Tuple[List[int], List[int]], **kwargs):
        self._base_ids = base_ids
        self._seq_ids = seq_ids
        self._max_len = 0
        super().__init__(**kwargs)

    @classmethod
    def _reconstruct(cls, distributed: bool, autocast: bool, log_dir: str, ckpt_dir: str, descr: str,
                     known_dim: int, unknown_dim: int, cat_dims: List[Tuple[int, int]], lae_trained: bool,
                     lae: nn.Module, optimizer_lae: Optimizer, lr_schd_lae: LRScheduler,
                     grad_scaler_lae: Optional[GradScaler], corr_mat: Optional[Tensor],
                     corr_mask: Optional[Tensor], mean: Optional[Tensor],
                     base_ids: Tuple[List[int], List[int]], seq_ids: Tuple[List[int], List[int]], max_len: int,
                     **kwargs) -> "SeriesTrainer":
        base = super()._reconstruct(
            distributed, autocast, log_dir, ckpt_dir, descr,
            known_dim, unknown_dim, cat_dims, lae_trained,
            lae, optimizer_lae, lr_schd_lae, grad_scaler_lae,
            corr_mat, corr_mask, mean
        )
        base.__class__ = cls
        base._base_ids, base._seq_ids = base_ids, seq_ids
        base._max_len = max_len
        return base

    def __reduce__(self):
        _, var = super().__reduce__()
        return self._reconstruct, var + (
            self._base_ids, self._seq_ids, self._max_len
        )

    def _make_dataloader(self, known: List[Tensor], unknown: List[Tensor], batch_size: int, shuffle: bool = True) -> \
            DataLoader:
        dataset = _TensorListDataset(known, unknown)
        sampler = DistributedSampler(dataset, shuffle=shuffle) if self._distributed else \
            RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True,
                                collate_fn=self._collate_fn)
        return dataloader

    def _make_infer_dataloader(self, known: List[Tensor], batch_size: int, shuffle: bool = True) -> DataLoader:
        dataset = _TensorListDataset(known)
        sampler = DistributedSampler(dataset, shuffle=shuffle) if self._distributed else \
            RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True,
                                collate_fn=self._collate_fn_infer)
        return dataloader

    def _collate_fn(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        all_known, all_unknown = [], []
        max_len = max(y.shape[0] for x, y in batch)
        for known, unknown in batch:
            known = known.expand(max_len, -1)
            all_known.append(known)

            placeholder = torch.zeros(max_len, unknown.shape[-1] + 1, device=unknown.device, dtype=torch.float32)
            placeholder[:unknown.shape[0], :-1] = unknown
            placeholder[:unknown.shape[0], -1] = 1
            all_unknown.append(placeholder)
        self._max_len = max(max_len, self._max_len)
        return torch.stack(all_known), torch.stack(all_unknown)

    def _collate_fn_infer(self, batch: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
        all_known = []
        for known, in batch:
            all_known.append(known.expand(self._max_len, -1))
        return torch.stack(all_known),
