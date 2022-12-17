"""Base trainer for degree prediction."""
import os
from abc import ABC
from typing import Optional, List

import pandas as pd
from torch import Tensor

from ..schema.database.base import ForeignKey
from ..schema import Table, SyntheticTable, Database, SyntheticDatabase


class DegreeTrainer(ABC):
    """Base trainer for degree prediction."""
    def __init__(self, foreign_keys: List[ForeignKey], descr: str = '', cache_dir: str = 'cached'):
        """
        **Args**:

        - `foreign_keys` (`List[ForeignKey]`): Foreign keys for this table's degree prediction.
        - `descr` (`str`): This degree trainer's short description.
        - `cache_dir` (`str`): Cache directory for information in this trainer.
        """
        self._foreign_keys = foreign_keys
        self._descr = descr
        self._cache_dir = cache_dir
        self._degrees_per_fk = []
        os.makedirs(descr, exist_ok=True)

    def fit(self, data: Table, context: Database):
        """
        Fit the degree trainer.

        **Args**:

        - `data` (`Table`): The table to predict degree on.
        - `context` (`Database`): The entire database.
        """
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
        self._fit(data, context)

    def _fit(self, data: Table, context: Database):
        raise NotImplementedError()

    def _get_scaling(self, scaling: Optional[List[float]]) -> List[float]:
        if scaling is None:
            return [1 for _ in self._foreign_keys]
        else:
            return scaling

    def _do_scaling(self, degrees: pd.Series, scaling: List[float], deg_known: pd.DataFrame, tolerance: float) -> pd.Series:
        assert len(scaling) == len(self._foreign_keys), \
            f'Number of scaling factors provided should be the same as the number of foreign keys. ' \
            f'Got {len(scaling)} factors but {len(self._foreign_keys)} foreign keys.'
        assert len(degrees) == len(deg_known), \
            f'Size of degrees predicted and the known part of degree table should be the same. ' \
            f'Got {len(degrees)} predicted degrees but {len(deg_known)} rows in the known part of degree table.'
        deg_known.loc[:, ('', 'degree_raw')] = degrees * np.prod(scaling)
        deg_known.loc[:, ('', 'degree')] = deg_known[('', 'degree_raw')].apply(round) \
            .apply(lambda x: x if x >= 0 else 0)

        for _ in range(self._max_scaling_iter):
            violated = False
            for fk, factor, real_degrees in zip(self._foreign_keys, scaling, self._degrees_per_fk):
                grouped = deg_known.groupby(fk.left, dropna=False, sort=False)[[('', 'degree'), ('', 'degree_raw')]]
                degrees_for_this_fk = grouped.sum()
                expected_mean = real_degrees.mean() * factor
                l, r = expected_mean * (1 - tolerance), expected_mean * (1 + tolerance)
                actual_mean = degrees_for_this_fk[('', 'degree')].mean()
                if l <= actual_mean <= r:
                    continue

                violated = True
                ratio = expected_mean / actual_mean
                for fk_val, deg_val in grouped:
                    int_deg = deg_val[('', 'degree')]
                    rounding = deg_val[('', 'degree_raw')]
                    expected_diff = int_deg.sum() * ratio - int_deg.sum()
                    if expected_diff < 0:
                        rounding = -rounding
                    offset = -rounding.min() + 0.01 if rounding.min() < 0 else 0
                    p = rounding + offset
                    indices = np.random.choice(range(len(int_deg)), round(expected_diff), p=p / p.sum())
                    for idx in indices:
                        if expected_diff > 0:
                            deg_known.loc[int_deg.iloc[idx].name, ('', 'degree')] += 1
                        else:
                            deg_known.loc[int_deg.iloc[idx].name, ('', 'degree')] -= 1
                deg_known.loc[:, ('', 'degree')] = deg_known[('', 'degree')].apply(lambda x: x if x >= 0 else 0)
            if not violated:
                break
        return deg_known[('', 'degree')]

    def predict(self, data: SyntheticTable, context: SyntheticDatabase, scaling: Optional[List[float]],
                tolerance: float = 0.05) -> (Tensor, pd.DataFrame):
        """
        Predict the degrees and constructed partially known augmented table for unknown part generation.

        **Args**:

        - `data` (`SyntheticTable`): The table to predict degree on.
        - `context` (`SyntheticDatabase`): The entire predicted database so far.
        - `scaling` (`Optional[List[float]]`): The scaling factors per foreign key.
          If not provided, all factors are set to 1.
          The factors given should correspond to the given foreign keys, in that corresponding order.
        - `tolerance` (`float`): Tolerance of degree prediction error in the expected degrees.

        **Return**: The known part of augmented table, normalized and original.
        """
        raise NotImplementedError()

