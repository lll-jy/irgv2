"""Predict degrees as if partial tabular."""
import math
import os.path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .base import DegreeTrainer
from ..tabular import create_trainer, TabularTrainer
from ..schema import Table, SyntheticTable, Database, SyntheticDatabase
from ..schema.database.base import ForeignKey


class DegreeAsTabularTrainer(DegreeTrainer):
    """Degree prediction trainer as if partial tabular."""
    def __init__(self, foreign_keys: List[ForeignKey], descr: str, cache_dir: str = 'cached',
                 **kwargs):
        super().__init__(foreign_keys, descr, cache_dir)
        self._train_kwargs = {
            n: v for n, v in kwargs.items()
            if n in {'epochs', 'batch_size', 'shuffle', 'save_freq', 'resume'}
        }
        self._trainer_kwargs = {
            n: v for n, v in kwargs.items()
            if n not in self._train_kwargs
        }
        self._trainer: Optional[TabularTrainer] = None
        self._max_scaling_iter = max_scaling_iter

    def _fit(self, data: Table, context: Database):
        deg_known, deg_unknown, cat_dims = data.deg_data()
        known_dim, unknown_dim = deg_known.shape[1], deg_unknown.shape[1]
        self._trainer = create_trainer(
            cat_dims=cat_dims, known_dim=known_dim, unknown_dim=unknown_dim,
            in_dim=known_dim, out_dim=unknown_dim,
            log_dir=os.path.join(self._cache_dir, 'tflog'),
            ckpt_dir=os.path.join(self._cache_dir, 'checkpoints'),
            descr=self._descr,
            **self._trainer_kwargs
        )
        self._trainer.train(deg_known, deg_unknown, **self._train_kwargs)

    def predict(self, data: SyntheticTable, context: SyntheticDatabase, scaling: Optional[List[float]],
                tolerance: float = 0.05) \
            -> (Tensor, pd.DataFrame):
        scaling = self._get_scaling(scaling)
        os.makedirs(os.path.join(self._cache_dir, 'deg_temp'), exist_ok=True)
        deg_cnt = 0
        known_tensors, augmented = [], []
        while not context.deg_finished(data.name):
            path = os.path.join(self._cache_dir, 'deg_temp', f'set{deg_cnt}.pt')
            if os.path.exists(path):
                known_tab, aug_tab = torch.load(path)
            else:
                known, expected_size = context.degree_known_for(data.name)
                known_original = data.data('degree')
                deg_tensor = self._trainer.inference(known).output[:, -self._trainer.unknown_dim].cpu()
                degrees = data.inverse_transform_degrees(deg_tensor)
                degrees = self._do_scaling(degrees, scaling, known_original, tolerance)
                data.assign_degrees(degrees)
                known_tab, _, _ = data.ptg_data()
                aug_tab = data.data('augmented')
                torch.save((known_tab, aug_tab), path)
            deg_cnt += 1
            known_tensors.append(known_tab)
            augmented.append(aug_tab)
        known_tab = torch.cat(known_tensors)
        augmented = pd.concat(augmented)
        return known_tab, augmented
