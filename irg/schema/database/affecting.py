"""
Affecting relation based augmenting mechanism.
All tables affecting the table are taken into account when generating it.
Please refer to [the paper](TODO: link) for detailed definition of `affect`.
"""
import logging
import os.path
from collections import defaultdict
from typing import Any, DefaultDict, List, Set, Tuple, Dict, Optional
from types import FunctionType

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import Tensor

from ..table import Table, SyntheticTable
from ..attribute import BaseAttribute, RawAttribute
from .base import Database, SyntheticDatabase, ForeignKey

_LOGGER = logging.getLogger()


def _q5(x):
    return x.quantile(0.05)


def _q10(x):
    return x.quantile(0.1)


def _q20(x):
    return x.quantile(0.2)


def _q25(x):
    return x.quantile(0.25)


def _q30(x):
    return x.quantile(0.3)


def _q40(x):
    return x.quantile(0.4)


def _q50(x):
    return x.quantile(0.5)


def _q60(x):
    return x.quantile(0.6)


def _q70(x):
    return x.quantile(0.7)


def _q75(x):
    return x.quantile(0.75)


def _q80(x):
    return x.quantile(0.8)


def _q90(x):
    return x.quantile(0.9)


def _q95(x):
    return x.quantile(0.95)


_quantile_func: Dict[str, FunctionType] = {
    'q5': _q5,
    'q10': _q10,
    'q20': _q20,
    'q25': _q25,
    'q30': _q30,
    'q40': _q40,
    'q50': _q50,
    'q60': _q60,
    'q70': _q70,
    'q75': _q75,
    'q80': _q80,
    'q90': _q90,
    'q95': _q95,
}


class AffectingDatabase(Database):
    """
    Database with joining mechanism involving all relevant tables.
    """
    def __init__(self, max_context: int = np.inf, agg_func: Optional[List[str]] = None, **kwargs):
        """
        **Args**:

        - `max_context` (`int`): Maximum number of columns for context. Default is 500.
          It is suggested not to allow too wide context tables because of execution time and potential OOM.
        - `agg_func` (`Optional[List[str]]`): List aggregate functions in augmenting involving non-ancestor-descendant
          tables. Supported functions include the following (default is [mean]):
            - mean, std, min, max: mean, standard deviation, minimum, and maximum values.
            - q5, q10, q20, q25, q30, q40, q50, q60, q70, q75, q80, q90, q95: quantile values.
        - `kwargs`: Arguments for [`Database`](./base#irg.schema.database.base.Database).
        """
        self._descendants: DefaultDict[str, List[ForeignKey]] = defaultdict(list)
        self._agg_func = agg_func if agg_func is not None else ['mean']
        self._agg_func = [(_quantile_func[f] if f in _quantile_func else f) for f in self._agg_func]
        self._max_context = max_context
        super().__init__(**kwargs)

    @property
    def mtype(self) -> str:
        return 'affecting'

    def augment(self):
        for name, path in self.tables():
            table = Table.load(path)
            self._augment_table(name, table)
            table.save(path)

    def _augment_table(self, name: str, table: Table, row_range: (int, int) = (0, np.inf)):
        foreign_keys = self._foreign_keys[name]
        augmented = pd.concat({name: table.data()}, axis=1)
        l, r = row_range
        if r == np.inf:
            r = len(augmented)
        augmented = augmented.iloc[l:r]
        degree = augmented.copy()
        # print('start aug', augmented[(name, 'student_token')].head())
        # print(' this table ', name, 'knows', table._known_cols, self[name]._known_cols)
        id_cols, attributes, fk_cols, fk_attr = set(), {}, set(), {}
        for i, foreign_key in enumerate(foreign_keys):
            parent_name = foreign_key.parent
            self._descendants[parent_name].append(foreign_key)
            prefix = f'fk{i}:{parent_name}'
            data, new_ids, new_attr = self._descendant_joined(name, parent_name)

            data = pd.concat({prefix: data}, axis=1)
            left = foreign_key.left
            right = [(prefix, col) for _, col in foreign_key.right]
            augmented = augmented.merge(data, how='left', left_on=left, right_on=right)
            degree = degree.merge(data, how='outer', left_on=left, right_on=right)

            id_cols |= {(prefix, col) for col in new_ids}
            attributes |= {(prefix, name): attr for name, attr in new_attr.items()}
            fk_cols |= {col for _, col in foreign_key.left}
            fk_attr |= {l_col: new_attr[r_col] for l_col, r_col in foreign_key.ref}

        aug_id_cols, deg_id_cols = set(), set()
        aug_attr, deg_attr = {}, {}
        for attr_name, attr in table.attributes().items():
            if attr.atype == 'id':
                aug_id_cols.add((name, attr_name))
                if attr_name in fk_cols:
                    deg_id_cols.add((name, attr_name))
            if attr_name not in fk_cols:
                aug_attr[(name, attr_name)] = attr
            else:
                aug_attr[(name, attr_name)] = fk_attr[attr_name]
                deg_attr[(name, attr_name)] = fk_attr[attr_name]
        degree_to_drop = [
            (name, col) for col in table.columns
            if col not in fk_cols and (name, col) in degree
        ]
        table.augment(
            augmented=augmented,
            degree=degree.drop(columns=degree_to_drop),
            augmented_ids=aug_id_cols | id_cols, degree_ids=deg_id_cols | id_cols,
            augmented_attributes=aug_attr | attributes, degree_attributes=deg_attr | attributes
        )

    def _descendant_joined(self, curr_name: str, parent_name: str) -> \
            Tuple[pd.DataFrame, Set[str], Dict[str, BaseAttribute]]:
        data, new_ids, all_attr = self[parent_name].augmented_for_join()
        original_cols = data.columns
        new_attr = {}
        for i, foreign_key in enumerate(self._descendants[parent_name]):
            if foreign_key.child == curr_name:
                continue
            desc_data, desc_ids, desc_attr = self._descendant_joined(curr_name, foreign_key.child)
            col_to_join = [col for _, col in foreign_key.left]

            desc_normalized = []
            for col in desc_data.columns:
                if col not in col_to_join:
                    col_normalized = desc_attr[col].transform(desc_data[col])
                    col_normalized = col_normalized.set_axis([f'{col}:{nc}' for nc in col_normalized.columns], axis=1)
                    desc_normalized.append(col_normalized)
            desc_normalized = pd.concat(desc_normalized, axis=1)
            desc_normalized[col_to_join] = desc_data[col_to_join]

            desc_agg = desc_normalized.groupby(by=col_to_join, dropna=False, sort=False)\
                .aggregate(func=self._agg_func).reset_index()
            if len(desc_agg.columns) - len(col_to_join) > self._max_context:
                pca = PCA(self._max_context)
                desc_agg_reduced = pca.fit_transform(desc_agg.drop(columns=col_to_join))
                desc_agg_reduced = pd.DataFrame(desc_agg_reduced,
                                                columns=[f'reduced{i}' for i in range(self._max_context)])
                desc_agg_reduced[col_to_join] = desc_agg[col_to_join]
                desc_agg = desc_agg_reduced
                _LOGGER.info(f'Fitted PCA to reduce dimension for context from descendant of {parent_name}.')
            desc_agg = desc_agg.set_axis([f'desc{i}:{foreign_key.child}/{c}{":" if n else ""}{n}' for c, n in desc_agg.columns], axis=1)
            col_to_join = [f'desc{i}:{foreign_key.child}/{c}' for c in col_to_join]
            data = data.merge(desc_agg, how='inner',
                              left_on=[col for _, col in foreign_key.right],
                              right_on=col_to_join).drop(columns=col_to_join)

            col_to_join = set(col_to_join)
            for col in desc_agg.columns:
                if col not in col_to_join:
                    new_attr[col] = RawAttribute(col, desc_agg[col])

        if len(data.columns) - len(original_cols) > self._max_context:
            pca = PCA(self._max_context)
            reduced = pca.fit_transform(data.drop(columns=original_cols))
            new_attr = {
                f'desc:reduce_{i}': RawAttribute(f'desc:reduce_{i}', reduced[:, i])
                for i in range(self._max_context)
            }
            reduced = pd.DataFrame(reduced, columns=[*new_attr])
            reduced[original_cols] = data[original_cols]
            data = reduced
        all_attr.update(new_attr)
        return data, new_ids, all_attr

    @staticmethod
    def _update_cls(item: Any):
        item.__class__ = AffectingDatabase
        item._descendants = defaultdict(list)
        # item._agg_func = agg_func if agg_func is not None else  # TODO: as input
        item._agg_func = ['mean']
        item._agg_func = [_quantile_func[f] if f in _quantile_func else f for f in item._agg_func]
        item._max_context = np.inf


class SyntheticAffectingDatabase(AffectingDatabase, SyntheticDatabase):
    """
    Synthetic database for affecting augmenting mechanism.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._used = defaultdict(int)
        self._total_groups = defaultdict(int)

    def _temp_deg_potential(self, table_name: str):
        os.makedirs(os.path.join(self._temp_cache, 'temp_deg'), exist_ok=True)
        os.makedirs(os.path.join(self._temp_cache, 'temp_deg', table_name), exist_ok=True)
        return os.path.join(self._temp_cache, 'temp_deg', table_name)

    def degree_known_for(self, table_name: str) -> Optional[Tensor]:
        if self._used[table_name] >= self._total_groups[table_name] and self._used[table_name] > 0:
            return None
        if self._used[table_name] == 0 and not os.path.exists(
                os.path.join(self._temp_deg_potential(table_name), f'set{self._used[table_name]}.pt')):
            table = self[table_name]
            foreign_keys = self._foreign_keys[table_name]

            df = pd.DataFrame()
            for fk in foreign_keys:
                new_df = self[fk.parent].data()[[col for _, col in fk.ref]]\
                    .rename(columns={pc: mc for mc, pc in fk.ref})
                new_df.loc[len(new_df)] = [np.nan for _ in fk.ref]
                if df.empty:
                    df = new_df
                else:
                    df = df.merge(new_df, how='cross')
            table.augment(df, df, set(), set(), {}, {})
            table.replace_data(df, False)
            for i in range(0, len(df), 100000):
                self._augment_table(table_name, table, (i, i+100000))
                deg, _, _ = table.deg_data()
                torch.save(deg, os.path.join(self._temp_deg_potential(table_name), f'set{i//100000}.pt'))
                self._total_groups[table_name] += 1
            self._used[table_name] = 1
            return torch.load(os.path.join(self._temp_deg_potential(table_name), 'set0.pt'))
        res = torch.load(os.path.join(self._temp_deg_potential(table_name), f'set{self._used[table_name]}.pt'))
        self._used[table_name] += 1
        return res

