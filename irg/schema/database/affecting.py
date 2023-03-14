"""
Affecting relation based augmenting mechanism.
All tables affecting the table are taken into account when generating it.
Please refer to [the paper](TODO: link) for detailed definition of `affect`.
"""
import logging
import os.path
from collections import defaultdict
from typing import Any, DefaultDict, List, Literal, Set, Tuple, Dict, Optional
from types import FunctionType

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import Tensor

from ..table import Table
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
            table, _ = self._load_table(name)
            self._augment_table(name, table)
            table.save(path)

    def create_table(self, name: str, meta: Dict[str, Any]) -> Table:
        table = super().create_table(name, meta)
        self._augment_table(name, table)
        table.save(self._table_paths[name])
        return table

    def _augment_table(self, name: str, table: Table, row_range: (int, int) = (0, np.inf), aug: bool = True) -> int:
        augmented = pd.concat({name: table.data()}, axis=1)
        l, r = row_range
        if r == np.inf:
            r = len(augmented)
        augmented = augmented.iloc[l:r]
        r = len(augmented) + l
        degree = augmented.copy()
        foreign_keys = self._foreign_keys[name]
        deg_cols = []
        deg_empty = pd.DataFrame()
        for fk in foreign_keys:
            if aug and all(table.attr_for_deg(c) for _, c in fk.left):
                new_df = self[fk.parent].data()[[col for _, col in fk.ref]] \
                    .rename(columns={pc: mc for mc, pc in fk.ref})
                deg_cols += [*new_df.columns]
                if deg_empty.empty:
                    deg_empty = new_df
                else:
                    deg_empty = deg_empty.merge(new_df, how='cross')
        degree = degree[[(name, col) for col in deg_cols]].drop_duplicates().reset_index(drop=True)
        if aug and not deg_empty.empty:
            deg_empty = deg_empty.merge(degree[name], how='left', indicator=True, on=deg_cols)
            deg_empty = deg_empty[deg_empty['_merge'] == 'left_only'].drop(columns=['_merge'])
            deg_empty = deg_empty.sample(min(len(deg_empty), 5 * len(degree)))
            deg_empty = pd.concat({name: deg_empty}, axis=1)
            degree = pd.concat([degree, deg_empty])

        id_cols, aug_attributes, deg_attributes, fk_cols, fk_attr = set(), {}, {}, set(), {}
        for i, foreign_key in enumerate(foreign_keys):
            parent_name = foreign_key.parent
            self._descendants[parent_name].append(foreign_key)
            prefix = f'fk{i}:{parent_name}'
            data, new_ids, new_attr = self._descendant_joined(name, parent_name)

            data = pd.concat({prefix: data}, axis=1)
            left = foreign_key.left
            right = [(prefix, col) for _, col in foreign_key.right]
            augmented = augmented.merge(data, how='left', left_on=left, right_on=right)
            if all(table.attr_for_deg(c) for _, c in foreign_key.left):
                degree = degree.merge(data, how='left', left_on=left, right_on=right)
                deg_attributes |= {(prefix, name): attr for name, attr in new_attr.items()}

            id_cols |= {(prefix, col) for col in new_ids}
            aug_attributes |= {(prefix, name): attr for name, attr in new_attr.items()}
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
            augmented_attributes=aug_attr | aug_attributes, degree_attributes=deg_attr | deg_attributes
        )
        return r - l

    def update_columns(self, name: str):
        super().update_columns(name)
        foreign_keys = self._foreign_keys[name]
        for fk in foreign_keys:
            self._descendants[fk.parent].append(fk)


    def _descendant_graph_till(self, till: str) -> Dict[str, List[ForeignKey]]:
        last_child = None
        for child, foreign_keys in self._foreign_keys.items():
            if child == till:
                break
            last_child = child
        if last_child is None:
            return {till: []}
        previous_result = self._descendant_graph_till(last_child)
        for foreign_key in self._foreign_keys[till]:
            previous_result[foreign_key.parent].append(foreign_key)
        previous_result[till] = []
        return previous_result

    def augmented_till(self, name: str, till: str, normalized: bool = True,
                       with_id: Literal['none', 'this', 'inherit'] = 'this') -> pd.DataFrame:
        previous_descendants = self._descendants[name]
        self._descendants[name] = self._descendant_graph_till(self.prev_table_of(till))[name]
        data, ids, all_attr = self._descendant_joined(
            curr_name=till,
            parent_name=name,
            with_id='this' if with_id == 'none' else with_id
        )
        if with_id == 'none':
            data = data.drop(columns=[c for c in data.columns if c in ids])
        all_columns = set(data.columns)
        if normalized:
            normalized_data = {}
            for attr_name, attr in all_attr.items():
                if attr_name in all_columns:
                    normalized_data[attr_name] = attr.transform(data[attr_name])
            data = pd.concat(normalized_data, axis=1)
        self._descendants[name] = previous_descendants
        return data

    def _descendant_joined(self, curr_name: str, parent_name: str,
                           with_id: Literal['none', 'this', 'inherit'] = 'this') -> \
            Tuple[pd.DataFrame, Set[str], Dict[str, BaseAttribute]]:
        data, new_ids, all_attr = self[parent_name].augmented_for_join(with_id=with_id)
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
            desc_agg = desc_agg.set_axis(
                [f'desc{i}:{foreign_key.child}/{c}{":" if n else ""}{n}' for c, n in desc_agg.columns], axis=1)
            col_to_join = [f'desc{i}:{foreign_key.child}/{c}' for c in col_to_join]
            data = data.merge(desc_agg, how='left',
                              left_on=[col for _, col in foreign_key.right],
                              right_on=col_to_join).drop(columns=col_to_join)
            for c in desc_agg.columns:
                if c in col_to_join:
                    continue
                data.loc[:, c] = data[c].fillna(0 if pd.isnull(desc_agg[c].mean()) else desc_agg[c].mean())

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

    def _temp_table_path(self, table_name) -> str:
        os.makedirs(os.path.join(self._temp_cache, 'temp_deg'), exist_ok=True)
        return os.path.join(self._temp_cache, 'temp_deg', f'{table_name}.pt')

    def degree_known_for(self, table_name: str) -> (Tensor, int):
        table = self[table_name]
        all_base = True
        if not os.path.exists(self._temp_table_path(table_name)):
            foreign_keys = self._foreign_keys[table_name]

            df = pd.DataFrame()
            for fk in foreign_keys:
                parent_table = self[fk.parent]
                if parent_table.ttype != 'base':
                    all_base = False
                new_df = parent_table.data()[[col for _, col in fk.ref]] \
                    .rename(columns={pc: mc for mc, pc in fk.ref})
                if df.empty:
                    df = new_df
                else:
                    df = df.merge(new_df, how='cross')
            torch.save(df, self._temp_table_path(table_name))
        else:
            df = torch.load(self._temp_table_path(table_name))
        table.augment(df, df, set(), set(), {}, {})
        table.replace_data(df, False)

        base = (self._used[table_name] - 1) * 100000
        size = self._augment_table(table_name, table, (base, base+100000), False)
        deg, _, _ = table.deg_data()
        real_size = len(self._real[table_name].data())
        self[table_name] = table
        if all_base:
            return None, None
        return deg, size / len(df) * real_size

    def deg_finished(self, table_name: str) -> bool:
        if not os.path.exists(self._temp_table_path(table_name)):
            self._used[table_name] += 1
            return False
        base = self._used[table_name] * 100000
        df = torch.load(self._temp_table_path(table_name))
        self._used[table_name] += 1
        return base > len(df)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._augment_table(key, value)
        aug_df = pd.read_pickle(self[key]._augmented_path())
        cols = [col for col in aug_df.columns if 'student_token' in col or 'module_code' in col]

    def save_dummy(self, table_name: str, table: Table):
        super().__setitem__(table_name, table)
