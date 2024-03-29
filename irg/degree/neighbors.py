import os.path
from collections import defaultdict
from typing import Optional, Dict

import pandas as pd
import torch
from sklearn.metrics.pairwise import euclidean_distances
from torch import Tensor
from tqdm import tqdm

from .base import DegreeTrainer
from ..schema import Table, Database, SyntheticTable, SyntheticDatabase
from ..utils.io import pd_to_pickle, pd_read_compressed_pickle


class DegreeFromNeighborsTrainer(DegreeTrainer):
    """Degree prediction by same degree values from nearest neighbor in real data."""
    def __init__(self, k: int = 10, **kwargs):
        """
        **Args**:

        - `k` (`int`): Number of neighbors with the highest scores to retain. Default is 10.
        """
        super().__init__(**kwargs)
        self._comb_counts = defaultdict(int)
        self._context = []
        self._k = k

    def _fit(self, data: Table, context: Database):
        deg_data = data.data('degree')
        self._comb_counts, self._context = defaultdict(int), []
        do_comb_count = True
        if os.path.exists(os.path.join(self._cache_dir, 'comb_counts.pt')):
            comb_counts = torch.load(os.path.join(self._cache_dir, 'comb_counts.pt'))
            self._comb_counts.update(comb_counts)
            do_comb_count = False
            torch.save(self._comb_counts, os.path.join(self._cache_dir, 'comb_counts.pt'))

        index_in_parent = []
        for i, fk in enumerate(self._foreign_keys):
            if fk.child != data.name:
                continue
            filepath = os.path.join(self._cache_dir, f'context{i:2d}.pt')
            parent_name = fk.parent
            parent_table = context[parent_name]
            if os.path.exists(filepath) and False:
                parent_normalized = pd_read_compressed_pickle(filepath)
            else:
                parent_normalized = context.augmented_till(parent_name, data.name, with_id='none')
                pd_to_pickle(parent_normalized, filepath)
            self._context.append(parent_normalized)

            if do_comb_count:
                parent_data = parent_table.data()[[r for l, r in fk.right]]
                assert len(parent_data) == len(parent_normalized)
                indices = []
                for _, row in tqdm(deg_data.iterrows(), total=len(deg_data), desc=f'Fit {data.name} deg {i} nb'):
                    deg_val = row[fk.left]
                    matches = parent_data.apply(lambda r: all(deg_val.values == r.values), axis=1)
                    assert sum(matches) == 1, f'Matched {sum(matches)}, but expected 1'
                    index = matches.astype('int').argmax()
                    indices.append(index)
                index_in_parent.append(indices)

        if do_comb_count:
            for comb, deg in zip(zip(*index_in_parent), deg_data[('', 'degree')]):
                self._comb_counts[comb] += int(deg)
        torch.save(self._comb_counts, os.path.join(self._cache_dir, 'comb_counts.pt'))

    def predict(self, data: SyntheticTable, context: SyntheticDatabase, scaling: Optional[Dict[str, float]],
                tolerance: float = 0.05) -> (Tensor, pd.DataFrame):
        index_correspondences = []
        parent_tables = []
        for fk, real_ctx in zip(self._foreign_keys, self._context):
            parent_name = fk.parent
            parent_normalized = context.augmented_till(parent_name, self._name, with_id='none')
            pairwise_euclidean = euclidean_distances(real_ctx, parent_normalized)
            # pairwise_euclidean = euclidean_distances(parent_normalized, real_ctx)
            values, correspondence = torch.topk(torch.tensor(pairwise_euclidean), self._k, dim=-1)
            rand = torch.argmax(torch.rand(*correspondence.shape) * values, dim=-1)
            correspondence = [row[i].item() for row, i in zip(correspondence, rand)]
            index_correspondences.append(correspondence)

            parent_normalized = context.augmented_till(parent_name, self._name, with_id='inherit', normalized=False)
            parent_tables.append(parent_normalized)

        pred_deg = []
        deg_known = pd.concat({data.name: pd.DataFrame()})
        for comb, cnt in self._comb_counts.items():
            pred_deg.append(cnt)
            new_row = {}
            for c, table, idx, (i, fk) in zip(comb, parent_tables, index_correspondences, enumerate(self._foreign_keys)):
                new_row[f'fk{i}:{fk.parent}'] = table.iloc[idx]
            new_row = pd.concat(new_row)
            deg_known = deg_known.append(new_row, ignore_index=True)
        for i, fk in enumerate(self._foreign_keys):
            for (pname, cname), l in zip(fk.right, fk.left):
                deg_known[l] = deg_known[(f'fk{i}:{pname}', cname)]
        data.save_degree_known(deg_known)
        pred_deg = self._do_scaling(
            degrees=pd.Series(pred_deg),
            scaling=scaling,
            deg_known=deg_known,
            tolerance=tolerance
        )

        total = 1
        for s in parent_tables:
            total *= len(s)
        real = pd.Series([*self._comb_counts.values()] + [0] * (total - len(self._comb_counts)))
        left = real.sum() * 0.9
        right = real.sum() * 1.1
        if not left <= pred_deg.sum() <= right:
            raise ValueError()
        data.assign_degrees(pred_deg)
        known_tab, _, _ = data.ptg_data()
        augmented = data.data('augmented')
        return known_tab, augmented
