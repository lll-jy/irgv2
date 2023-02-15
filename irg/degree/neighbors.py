import os.path
from collections import defaultdict
from typing import Optional, Dict

import numpy as np
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._comb_counts = defaultdict(int)
        self._context = []

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
                parent_normalized = context.augmented_till(parent_name, data.name)
                # if parent_table.is_independent():
                #     parent_normalized = pd.concat({parent_name: parent_table.data(normalize=True, with_id='none')})
                # else:
                #     parent_normalized = parent_table.data('augmented', normalize=True, with_id='none')
                pd_to_pickle(parent_normalized, filepath)
                print('!!! re do', self._name, parent_name, parent_normalized.shape, flush=True)
                print(parent_normalized.columns.tolist(), flush=True)
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
            for comb in zip(*index_in_parent):
                self._comb_counts[comb] += 1
        torch.save(self._comb_counts, os.path.join(self._cache_dir, 'comb_counts.pt'))

    def predict(self, data: SyntheticTable, context: SyntheticDatabase, scaling: Optional[Dict[str, float]],
                tolerance: float = 0.05) -> (Tensor, pd.DataFrame):
        index_correspondences = []
        parent_tables = []
        for fk, real_ctx in zip(self._foreign_keys, self._context):
            parent_name = fk.parent
            # print('predict for', fk.parent, fk.child)
            parent_table = context[parent_name]
            if parent_table.is_independent():
                parent_normalized = pd.concat({parent_name: parent_table.data(normalize=True, with_id='none')})
            else:
                parent_data = parent_table.data('augmented')
                parent_normalized = parent_table.data('augmented', normalize=True, with_id='none')
                assert len(parent_data) == len(parent_normalized)
            pairwise_euclidean = euclidean_distances(parent_normalized, real_ctx)
            correspondence = torch.topk(torch.tensor(pairwise_euclidean), 10, dim=-1).indices
            rand = torch.argmax(torch.rand(*correspondence.shape), dim=-1)
            correspondence = [row[i].item() for row, i in zip(correspondence, rand)]
            # correspondence = correspondence[torch.argmax(rand, dim=-1)]
            # is_max_indicator = pairwise_euclidean == pairwise_euclidean.max(axis=-1).reshape(-1, 1)
            # correspondence = [np.random.choice(np.flatnonzero(row)) for row in is_max_indicator]
            index_correspondences.append(correspondence)

            # raw_parent = parent_table.data(with_id='this')
            # for col_name in raw_parent.columns:
            #     if not any(col_name == x and y != 'is_nan' for x, y in parent_normalized.columns):
            #         parent_normalized[(col_name, '')] = raw_parent[col_name]
            # parent_tables.append(parent_normalized)
            # parent_table = self._real.augmented_till(parent_name, )
            if parent_table.is_independent():
                parent_tables.append(pd.concat({parent_name: parent_table.data(with_id='this')}))
            else:
                parent_tables.append(parent_table.data(variant='augmented', with_id='this'))

        pred_deg = []
        deg_known = pd.concat({data.name: pd.DataFrame()})
        for comb in zip(*index_correspondences):
            pred_deg.append(self._comb_counts[comb])
            new_row = {}
            for c, table_normalized, (i, fk) in zip(comb, parent_tables, enumerate(self._foreign_keys)):
                new_row[f'fk{i}:{fk.parent}'] = table_normalized.iloc[c]
            new_row = pd.concat(new_row)
            if len(pred_deg) == 1:
                print(new_row, flush=True)
            deg_known = deg_known.append(new_row, ignore_index=True)
        for i, fk in enumerate(self._foreign_keys):
            for (pname, cname), l in zip(fk.right, fk.left):
                deg_known[l] = deg_known[(f'fk{i}:{pname}', cname)]
        data.save_degree_known(deg_known)
        pred_deg = self._do_scaling(
            degrees=pd.Series(pred_deg),
            scaling=scaling,
            deg_known=deg_known,
            # deg_known=data.data('degree') if not data.is_independent() else pd.concat({data.name: pd.DataFrame()}),
            tolerance=tolerance
        )

        print('pred deg', data.name, pred_deg.sum(), pred_deg.describe())
        data.assign_degrees(pred_deg)
        print('call fake')
        known_tab, _, _ = data.ptg_data()
        augmented = data.data('augmented')
        real_this = context._real[data.name]
        print('call real')
        k, _, _ = real_this.ptg_data()
        print('result deg', data.name, '\n',
              'knownshape', known_tab.shape, deg_known.shape, k.shape, '\n',
              augmented.shape, real_this.data('augmented')[augmented.columns].shape)
        print(augmented.columns.tolist(), flush=True)
        return known_tab, augmented
