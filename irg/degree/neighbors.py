from typing import Optional, List

import pandas as pd
from torch import Tensor

from .base import DegreeTrainer
from ..schema import Table, Database, SyntheticTable, SyntheticDatabase


class DegreeFromNeighborsTrainer(DegreeTrainer):
    """Degree prediction by same degree values from nearest neighbor in real data."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rows_per_fk = []
        self._context = []

    def _fit(self, data: Table, context: Database):
        deg_data = data.data('degree')
        self._rows_per_fk, self._context = [], []
        for fk in self._foreign_keys:
            parent_name = fk.parent
            parent_table = context[parent_name]
            parent_data = parent_table.data('augmented')
            parent_normalized = parent_table.data('augmented', normalize=True, with_id='none')
            assert len(parent_data) == len(parent_normalized)
            self._context.append(parent_normalized)
            fk_cols = [(parent_name, col) for _, col in fk.right]
            rows_per_val = {}
            for fk_val, group_val in parent_data.groupby(fk_cols, sort=False, dropna=False):
                degrees = deg_data[deg_data[fk.left] == fk_val]
                rows_per_val[fk_val] = group_val.index
            self._rows_per_fk.append(rows_per_val)

    def predict(self, data: SyntheticTable, context: SyntheticDatabase, scaling: Optional[List[float]],
                tolerance: float = 0.05) -> (Tensor, pd.DataFrame):
        pass

