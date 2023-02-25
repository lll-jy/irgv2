"""Series tabular table data structure that holds data and metadata of tables in a database."""

from typing import Collection, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from torch import Tensor

from .table import Table


class SeriesTable(Table):
    def __init__(self, name: str, series_id: str, base_cols: Optional[Collection[str]] = None,
                 id_cols: Optional[Iterable[str]] = None, attributes: Optional[Dict[str, dict]] = None,
                 data: Optional[pd.DataFrame] = None,
                 # determinants: Optional[List[List[str]]] = None,
                 # formulas: Optional[Dict[str, str]] = None,
                 # temp_cache: str = '.temp'
                 **kwargs):
        if base_cols is None:
            base_cols = set()
        self._base_cols = base_cols | {'.series_degree'}
        if id_cols is None:
            id_cols = set()
        id_cols |= {series_id}
        if attributes is None:
            attributes = {}

        assert '.series_degree' not in attributes
        attributes['.series_degree'] = {
            'type': 'numerical',
            'min_val': 0,
            'name': '.series_degree'
        }

        # if data is not None:
        #     base_data = data[base_cols]
        #     base_data['.series_degree'] = 1
        #     base_data = base_data.groupby(base_cols, dropna=False, as_index=False).sum()
        # else:
        #     base_data = None

        self._index_groups = []
        super().__init__(name=name, ttype='series', id_cols=id_cols, attributes=attributes, **kwargs)

        # self._base = Table(
        #     name=f'{name}.base',
        #     id_cols=id_cols & set(base_cols),
        #     attributes={n: a for n, a in attributes.items() if n in base_cols},
        #     data=base_data,
        #     determinants=[det for det in determinants if det[0] in base_cols],
        #     temp_cache=f'{temp_cache}.base', **kwargs
        # )
        # self._series = Table(
        #     name=f'{name}.series',
        #     id_cols=(id_cols - set(base_cols)) | {series_id},
        #     attributes={n: a for n, a in attributes.items() if n not in base_cols},
        #     data=None if data is None else data.drop(columns=base_cols),
        #     determinants=[det for det in determinants if det[0] not in base_cols],
        #     temp_cache=f'{temp_cache}.series', **kwargs
        # )

    def fit(self, data: pd.DataFrame, force_redo: bool = False, **kwargs):
        data['.series_degree'] = 1
        for _, group in data.groupby(self._base_cols):
            data[group.index, '.series_degree'] = len(group)
            self._index_groups.append(group.index)
        super().fit(data, force_redo, **kwargs)

    def sg_data(self) -> Tuple[Tensor, Tensor, List[Tuple[int, int]],
                               Tuple[List[int], List[int]], Tuple[List[int], List[int]]]:
        known, unknown, cat_dims = self.ptg_data()
        base_known_ids, base_unknown_ids, seq_known_ids, seq_unknown_ids = [], [], [], []
        acc_known, acc_unknown = 0, 0
        for (table, attr_name), attr in self._augmented_attributes.items():
            is_known = table != self._name or attr_name in self._known_cols
            is_in_base = table == self._name and attr_name in self._base_cols
            attr_width = len(attr.transformed_columns)
            if is_known:
                if is_in_base:
                    base_known_ids.extend(range(acc_known, acc_known + attr_width))
                seq_known_ids.extend(range(acc_known, acc_known + attr_width))
                acc_known += attr_width
            else:
                if is_in_base:
                    base_unknown_ids.extend(range(acc_known, acc_known + attr_width))
                    seq_known_ids.extend(range(acc_known, acc_known + attr_width))
                else:
                    seq_unknown_ids.extend(range(acc_known, acc_known + attr_width))
                acc_unknown += attr_width

        all_known, all_unknown = [], []
        max_len = max(len(x) for x in self._index_groups)
        for group in self._index_groups:
            all_known.append(self._make_padded(max_len, known, group))
            all_unknown.append(self._make_padded(max_len, unknown, group))
        all_known = torch.stack(all_known)
        all_unknown = torch.stack(all_unknown)
        return all_known, all_unknown, cat_dims, (base_known_ids, base_unknown_ids), (seq_known_ids, seq_unknown_ids)

    @staticmethod
    def _make_padded(max_len: int, src: Tensor, group: Collection[int], col_ids: Optional[Collection[int]] = None):
        if col_ids is None:
            width = src.shape[-1]
        else:
            width = len(col_ids)
        placeholder = torch.zeros(max_len, width + 1, device=src.device, dtype=torch.float32)
        placeholder[:len(group), :width] = src[group, col_ids] if col_ids is not None else src[group]
        placeholder[:len(group), -1] = 1
        return placeholder
