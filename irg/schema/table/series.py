"""Series tabular table data structure that holds data and metadata of tables in a database."""
import os
from typing import Collection, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import Tensor

from .table import Table, SyntheticTable
from ...utils.errors import NotFittedError
from ...utils import SeriesInferenceOutput


class SeriesTable(Table):
    def __init__(self, name: str, series_id: str, base_cols: Optional[Collection[str]] = None,
                 id_cols: Optional[Iterable[str]] = None, attributes: Optional[Dict[str, dict]] = None,
                 **kwargs):
        if base_cols is None:
            base_cols = set()
        self._base_cols = base_cols
        self._series_id = series_id
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

        self._index_groups = []
        super().__init__(name=name, ttype='series', id_cols=id_cols, attributes=attributes, **kwargs)

    def fit(self, data: pd.DataFrame, force_redo: bool = False, **kwargs):
        data['.series_degree'] = 1
        for _, group in data.groupby(self._base_cols):
            data[group.index, '.series_degree'] = len(group)
            self._index_groups.append(group.sort_values(by=[self._series_id]).index)
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
            known_row = known[group].mean(dim=0)
            known_group = known_row.expand(max_len, -1)
            all_known.append(known_group)

            placeholder = torch.zeros(max_len, unknown.shape[-1], device=unknown.device, dtype=torch.float32)
            placeholder[:len(group)] = unknown[group]
            placeholder[:len(group), -1] = 1
            all_unknown.append(placeholder)
        all_known = torch.stack(all_known)
        all_unknown = torch.stack(all_unknown)
        return all_known, all_unknown, cat_dims, (base_known_ids, base_unknown_ids), (seq_known_ids, seq_unknown_ids)


class SyntheticSeriesTable(SeriesTable, SyntheticTable):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._real_cache = '.temp' if 'temp_cache' not in kwargs else kwargs['temp_cache']

    def _describer_path(self, idx: int) -> str:
        return os.path.join(self._real_cache, 'describers', f'describer{idx}.json')

    def _degree_attr_path(self) -> str:
        return os.path.join(self._real_cache, 'deg_attr.pkl')

    @classmethod
    def from_real(cls, table: SeriesTable, temp_cache: Optional[str] = None) -> "SyntheticSeriesTable":
        synthetic = SyntheticSeriesTable(
            name=table._name, ttype=table._ttype, need_fit=False,
            id_cols={*table._id_cols}, attributes=table._attr_meta,
            determinants=table._determinants, formulas=table._formulas,
            temp_cache=temp_cache if temp_cache is not None else table._temp_cache,
            series_id=table._series_id, base_cols=table._base_cols
        )
        return cls._copy_attributes(table, synthetic)

    @staticmethod
    def _copy_attributes(src: SeriesTable, target: "SyntheticSeriesTable") -> "SyntheticSeriesTable":
        super()._copy_attributes(src, target)
        target._index_groups = src._index_groups
        return target

    def inverse_transform(self, normalized_core: Union[Tensor, SeriesInferenceOutput], replace_content: bool = True) -> \
            pd.DataFrame:
        if not self._fitted:
            raise NotFittedError('Table', 'inversely transforming predicted synthetic data')
        if isinstance(normalized_core, SeriesInferenceOutput):
            lengths = normalized_core.lengths
            normalized_core = normalized_core.output
            regroup = False
        else:
            normalized_core = normalized_core.view(1, *normalized_core.shape)
            lengths = [normalized_core.shape[1]]
            regroup = True

        flattened = []
        for group, length in zip(normalized_core, lengths):
            group = group[:length, -self._unknown_dim:].cpu()
            flattened.append(group)

        flattened = torch.cat(flattened)
        columns, recovered_df = self._recover_core(flattened)

        if regroup:
            lengths = []
            new_data = []
            for _, data in recovered_df.groupby(self._base_cols, dropna=False):
                lengths.append(len(data))
                new_data.append(data)
            recovered_df = pd.concat(new_data, ignore_index=True).reset_index(drop=True)
        acc = 0
        for length in lengths:
            recovered_df.loc[acc:acc+length-1, self._series_id] = self._attributes[self._series_id]\
                .generate(length).tolist()

        return self._post_inverse_transform(recovered_df, columns, replace_content, flattened)
