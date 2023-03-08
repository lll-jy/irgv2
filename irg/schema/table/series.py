"""Series tabular table data structure that holds data and metadata of tables in a database."""
import os
from typing import Collection, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

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
        if attributes is None:
            attributes = {}

        assert all(x not in attributes for x in {'.series_degree', '.series_increase', '.group_id', '.series_base'})
        attributes['.series_degree'] = {
            'type': 'numerical',
            'min_val': 0,
            'name': '.series_degree'
        }

        series_id_attr = {} if series_id not in attributes else attributes[series_id]
        increase_type = 'numerical'
        if attributes.get(series_id) is not None and series_id_attr.get('type') is not None:
            if series_id_attr['type'] in {'datetime', 'timedelta'}:
                increase_type = 'timedelta'
            elif series_id_attr['type'] == 'id':
                increase_type = 'id'
            elif series_id_attr['type'] != 'mumerical':
                raise NotImplementedError(f'Attribute of type {series_id_attr["type"]} cannot be used '
                                          f'as series ID. Please use something which difference can be calculated.')
        attributes['.series_increase'] = {
            'type': increase_type,
            'min_val': '0',
            'name': '.series_increase'
        }
        attributes['.series_base'] = {
            **{k: v for k, v in series_id_attr.items() if k != 'name'},
            'name': '.series_base'
        }

        self._index_groups = []
        need_fit = True if 'need_fit' not in kwargs else kwargs['need_fit']
        kwargs['need_fit'] = False
        super().__init__(name=name, ttype='series', id_cols=id_cols, attributes=attributes, **kwargs)
        self._need_fit = need_fit
        if need_fit and kwargs.get('data') is not None:
            self.fit(kwargs.get('data'), **{k: v for k, v in kwargs.items() if k != 'data'})

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        new_data = []
        acc = 0
        assert not ({'.series_degree', '.series_increase', '.group_id', '.series_base'} & set(data.columns))
        if self._base_cols:
            grouped = data.groupby([*self._base_cols], as_index=False)
        else:
            grouped = [(i, x.to_frame().T) for i, x in data.iterrows()]
        for _, group in tqdm(grouped, desc=f'Grouping series {self._name}'):
            sorted_group = group.sort_values(by=[self._series_id]).reset_index(drop=True)
            sorted_group.loc[:, '.series_degree'] = len(group)
            sorted_series_id = sorted_group[self._series_id].values
            sorted_group.loc[:, '.series_increase'] = 0
            sorted_group.loc[:, '.series_base'] = sorted_series_id[0]
            sorted_group.loc[1:, '.series_increase'] = sorted_series_id[1:] - sorted_series_id[:-1]
            sorted_group.loc[:, '.group_id'] = len(new_data)
            new_data.append(sorted_group)
            self._index_groups.append(pd.Index(range(acc, acc + len(group))))
            acc += len(group)
        data = pd.concat(new_data)
        data = data.set_index('.group_id')
        return data

    def replace_data(self, new_data: pd.DataFrame, replace_attr: bool = True):
        new_data = self._preprocess_data(new_data)
        super().replace_data(new_data, replace_attr)

    def fit(self, data: pd.DataFrame, force_redo: bool = False, **kwargs):
        print('fit series', flush=True)
        do_group = True
        if os.path.exists(self._data_path()):
            loaded_data = pd.read_pickle(self._data_path())
            print('!!! exists', self._data_path(), os.stat(self._data_path()).st_mtime,
                  [*data.columns], flush=True)
            if {'.series_degree', '.series_increase', '.series_base'} <= set(data.columns):
                do_group = False
                data = loaded_data

        if do_group:
            data = self._preprocess_data(data)
            data.to_pickle(self._data_path())
            print('index!!', data.index, flush=True)
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
            lengths = [normalized_core.shape[1]]
            regroup = True

        columns, recovered_df, normalized_core = self._recover_core(normalized_core)

        if regroup:
            lengths = []
            new_data = []
            if self._base_cols:
                grouped = recovered_df.groupby([*self._base_cols], as_index=False, dropna=False)
            else:
                grouped = [(i, x.to_frame().T) for i, x in recovered_df.iterrows()]
            # for _, data in recovered_df.groupby([*self._base_cols], dropna=False):
            for _, data in tqdm(grouped, desc=f'Regrouping series {self._name}'):
                lengths.append(len(data))
                new_data.append(data)
            recovered_df = pd.concat(new_data, ignore_index=True).reset_index(drop=True)

        acc = 0
        for length in lengths:
            if self._series_id in self._id_cols:
                recovered_df.loc[acc:acc+length-1, self._series_id] = self._attributes[self._series_id]\
                    .generate(length).tolist()
            else:
                base = recovered_df.loc[acc:acc+length-1, '.series_base'].mean()
                recovered_df.loc[acc, self._series_id] = base
                for i in range(1, length):
                    recovered_df.loc[acc+i, self._series_id] = \
                        recovered_df.loc[acc+i-1, self._series_id] + recovered_df.loc[acc+i, '.series_increase']

        return self._post_inverse_transform(recovered_df, columns, replace_content, normalized_core)
