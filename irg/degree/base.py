"""Base trainer for degree prediction."""
import math
import os
from abc import ABC
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from ..schema.database.base import ForeignKey
from ..schema import Table, SyntheticTable, Database, SyntheticDatabase


class DegreeTrainer(ABC):
    """Base trainer for degree prediction."""
    def __init__(self, foreign_keys: List[ForeignKey], descr: str = '', cache_dir: str = 'cached',
                 max_scaling_iter: int = 10, unique: bool = False, **kwargs):
        """
        **Args**:

        - `foreign_keys` (`List[ForeignKey]`): Foreign keys for this table's degree prediction.
        - `descr` (`str`): This degree trainer's short description.
        - `cache_dir` (`str`): Cache directory for information in this trainer.
        - `max_scaling_iter` (`int`): Maximum number of iterations to run to adjust degrees to integers to a desired
          sum.
        - `unique` (`bool`): True if each combination of foreign keys appear at most once. Default is `False`.
        """
        self._foreign_keys = foreign_keys
        self._descr = descr
        self._cache_dir = cache_dir
        self._name = None
        self._max_scaling_iter = max_scaling_iter
        self._unique = unique
        os.makedirs(self._cache_dir, exist_ok=True)
        self._degrees_per_fk = []

    def fit(self, data: Table, context: Database):
        """
        Fit the degree trainer.

        **Args**:

        - `data` (`Table`): The table to predict degree on.
        - `context` (`Database`): The entire database.
        """
        self._name = data.name
        if os.path.exists(os.path.join(self._cache_dir, 'deg_per_fk.pt')):
            self._degrees_per_fk = torch.load(os.path.join(self._cache_dir, 'deg_per_fk.pt'))
        else:
            self._degrees_per_fk = []
            this_data = pd.concat({data.name: data.data()}, axis=1)
            for fk in self._foreign_keys:
                parent_name = fk.parent
                parent_table = context[parent_name]
                parent_data = pd.concat({parent_name: parent_table.data()}, axis=1)
                this_data_for_fk = this_data[fk.left]
                parent_data_for_fk = parent_data[fk.right]
                joined = parent_data_for_fk.merge(
                    this_data_for_fk, how='left',
                    left_on=fk.right, right_on=fk.left
                )
                degrees = joined.groupby(fk.left).size()
                self._degrees_per_fk.append(degrees)
            torch.save(self._degrees_per_fk, os.path.join(self._cache_dir, 'deg_per_fk.pt'))
        self._fit(data, context)

    def _fit(self, data: Table, context: Database):
        raise NotImplementedError()

    def _get_scaling(self, scaling: Optional[List[float]]) -> List[float]:
        if scaling is None:
            return [1 for _ in self._foreign_keys]
        else:
            return scaling

    @staticmethod
    def _round_sumrange(data: pd.Series, l: float, r: float, till_in_range: bool = False) -> (pd.Series, float):
        data = data.apply(lambda x: 0 if x < 0 else x)
        l = math.floor(l)
        r = math.ceil(r)
        smallest = data.apply(math.floor)
        if smallest.sum() >= r:
            if till_in_range:
                nd = math.ceil(smallest.sum() - r)
                while smallest.sum() > r:
                    to_del = np.random.choice(smallest[smallest >= 1].index, nd, replace=False)
                    for d in to_del:
                        smallest[d] -= 1
                return smallest, -nd
            return smallest, 0.
        largest = data.apply(math.ceil)
        if largest.sum() <= l:
            if till_in_range:
                na = math.ceil(l - largest.sum())
                to_add = np.random.choice(largest, na, replace=True)
                for a in to_add:
                    largest[a] += 1
                return largest, na
            return largest, 1.
        l_thres = 0
        r_thres = 1
        while True:
            m = (l_thres + r_thres) / 2
            current = data.apply(lambda x: math.floor(x) if x < math.floor(x) + m else math.ceil(x))
            curr_sum = current.sum()
            if l <= curr_sum <= r or r_thres - l_thres < 1e-6:
                return current, m
            if curr_sum < l:
                l_thres = m
            else:
                r_thres = m

    def _do_scaling(self, degrees: pd.Series, scaling: Dict[str, float], deg_known: pd.DataFrame, tolerance: float) \
            -> pd.Series:
        assert len(degrees) == len(deg_known), \
            f'Size of degrees predicted and the known part of degree table should be the same. ' \
            f'Got {len(degrees)} predicted degrees but {len(deg_known)} rows in the known part of degree table.'
        degrees = degrees.apply(lambda x: x if x > 0 else 0)
        scaling_factors = [scaling[fk.parent] for fk in self._foreign_keys]
        raw_degrees = degrees * np.prod(scaling_factors) * scaling[self._name]
        int_degrees = raw_degrees.apply(lambda x: round(x) if x >= 0 else 0)
        deg_known = pd.concat([deg_known, pd.DataFrame({
            ('', 'degree_raw'): raw_degrees,
            ('', 'degree'): int_degrees
        })], axis=1)

        for _ in range(self._max_scaling_iter):
            violated = False
            for fk, factor, real_degrees in zip(self._foreign_keys, scaling_factors, self._degrees_per_fk):
                grouped = deg_known.groupby(fk.left, dropna=False, sort=False)[[
                    ('', 'degree'), ('', 'degree_raw')
                ]]
                degrees_for_this_fk = grouped.transform('sum')
                expected_mean = real_degrees.mean() * factor
                l, r = expected_mean * (1 - tolerance), expected_mean * (1 + tolerance)
                scaled_deg, thres = self._round_sumrange(degrees_for_this_fk[('', 'degree')],
                                                         l * len(degrees_for_this_fk), r * len(degrees_for_this_fk))
                deg_known[('', 'degree')] = deg_known[('', 'degree')].apply(
                    lambda x: math.floor(x) if x < thres + math.floor(x) else math.ceil(x))
                grouped = deg_known.groupby(fk.left, dropna=False, sort=False)[[
                    ('', 'degree'), ('', 'degree_raw')
                ]]
                actual_mean = scaled_deg.mean()
                if l <= actual_mean <= r:
                    continue

                violated = True
                expected_std = (real_degrees * factor).std()
                actual_std = scaled_deg.std()

                def renorm(x):
                    x = (x - actual_mean) / max(actual_std, 1e-5)
                    return x * expected_std + expected_mean
                total_change = 0
                fake_total_change = 0
                for fk_val, deg_val in grouped:
                    rounding = renorm(deg_val[('', 'degree_raw')])
                    rsum = rounding.sum()
                    suml, sumr = rsum * (1 - tolerance), rsum * (1 - tolerance)
                    int_deg, val_thres = self._round_sumrange(
                        renorm(deg_val[('', 'degree')]), suml, sumr
                    )
                    deg_known.loc[int_deg.index, ('', 'degree')] = int_deg
                    deg_known.loc[int_deg.index, ('', 'degree_raw')] = rounding
                    if suml <= int_deg.sum() <= sumr:
                        continue
                    int_sum = int_deg.sum()
                    suml, sumr = math.floor(suml), math.ceil(sumr)
                    expected_diff = (sumr - int_sum) if int_sum > sumr else (int_sum - suml)
                    if expected_diff < 0:
                        rounding = -rounding
                    offset = -rounding.min() + 0.1 if rounding.min() < 0 else 0.1
                    p = rounding + offset
                    indices = np.random.choice(range(len(int_deg)), round(abs(expected_diff)), p=p / p.sum())
                    for idx in indices:
                        fake_total_change += 1
                        if expected_diff > 0:
                            if not self._unique or deg_known.loc[int_deg.index[idx], ('', 'degree')] <= 1:
                                total_change += 1
                            deg_known.loc[int_deg.index[idx], ('', 'degree')] += 1
                        else:
                            if deg_known.loc[int_deg.index[idx], ('', 'degree')] > 0:
                                total_change -= 1
                            deg_known.loc[int_deg.index[idx], ('', 'degree')] -= 1
                deg_known.loc[:, ('', 'degree')] = deg_known[('', 'degree')].apply(lambda x: x if x >= 0 else 0)
            if not violated:
                break
        return deg_known[('', 'degree')]

    def predict(self, data: SyntheticTable, context: SyntheticDatabase, scaling: Optional[Dict[str, float]],
                tolerance: float = 0.05) -> (Tensor, pd.DataFrame):
        """
        Predict the degrees and constructed partially known augmented table for unknown part generation.

        **Args**:

        - `data` (`SyntheticTable`): The table to predict degree on.
        - `context` (`SyntheticDatabase`): The entire predicted database so far.
        - `scaling` (`Optional[Dict[str, float]]`): The scaling factors per foreign key.
          If not provided, all factors are set to 1.
          The factors given should correspond to the given foreign keys, in that corresponding order.
        - `tolerance` (`float`): Tolerance of degree prediction error in the expected degrees.

        **Return**: The known part of augmented table, normalized and original.
        """
        raise NotImplementedError()

