from collections import defaultdict
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from torch import Tensor

from .base import DegreeTrainer
from ..schema import Table, Database, SyntheticTable, SyntheticDatabase


class DegreeFromNeighborsTrainer(DegreeTrainer):
    """Degree prediction by same degree values from nearest neighbor in real data."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._comb_counts = defaultdict(int)
        self._context = []

    def _fit(self, data: Table, context: Database):
        deg_data = data.data('degree')
        self._comb_counts, self._context = defaultdict(int), []
        index_in_parent = []
        for fk in self._foreign_keys:
            parent_name = fk.parent
            parent_table = context[parent_name]
            parent_normalized = parent_table.data('augmented', normalize=True, with_id='none')
            assert len(parent_data) == len(parent_normalized)
            self._context.append(parent_normalized)

            parent_data = parent_table.data()[fk.right]
            indices = []
            for i, row in deg_data.iterrows():
                deg_val = row[fk.left]
                matches = parent_data.apply(lambda row: row == deg_val, axis=1)
                assert sum(matches) == 1
                index = matches.argmax()
                indices.append(index)
            index_in_parent.append(indices)

        for comb in zip(*index_in_parent):
            self._comb_counts[comb] += 1

    def predict(self, data: SyntheticTable, context: SyntheticDatabase, scaling: Optional[List[float]],
                tolerance: float = 0.05) -> (Tensor, pd.DataFrame):
        index_correspondences = []
        for fk, real_ctx in zip(self._foreign_keys, self._context):
            parent_name = fk.parent
            parent_table = context[parent_name]
            parent_data = parent_table.data('augmented')
            parent_normalized = parent_table.data('augmented', normalize=True, with_id='none')
            assert len(parent_data) == len(parent_normalized)
            pairwise_euclidean = euclidean_distances(parent_normalized, real_ctx)
            is_max_indicator = pairwise_euclidean == pairwise_euclidean.max(axis=-1)
            correspondence = [np.random.choice(np.flatnonzero(row)) for row in is_max_indicator]
            index_correspondences.append(correspondence)

        pred_deg = []
        for comb in zip(*index_correspondences):
            pred_deg.append(self._comb_counts[comb])
        pred_deg = self._do_scaling(pd.Series(pred_deg))

        data.assign_degrees(pred_deg)
        known_tab, _, _ = data.ptg_data()
        augmented = data.data('augmented')
        return known_tab, augmented
