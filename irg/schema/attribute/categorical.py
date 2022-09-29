"""Handler for categorical data."""

import logging
import os
import pickle
from typing import Optional, List, Tuple, Dict, Collection

import pandas as pd

from .base import BaseAttribute, BaseTransformer
from ...utils.io import pd_to_pickle
from ...utils.dist import fast_map_dict
from ...utils import pd_mp

_LOGGER = logging.getLogger()


class CategoricalTransformer(BaseTransformer):
    """
    Transformer for categorical data.

    The transformed columns (after `is_nan`) are `cat_0`, `cat_1`, ..., `cat_k`
    for an attribute with k+1 categories.
    """
    def __init__(self, temp_cache: str = '.temp'):
        super().__init__(temp_cache)
        self._label2id, self._id2label, self._cat_cnt = None, None, 0

    def _save_additional_info(self):
        with open(os.path.join(self._temp_cache, 'info.pkl'), 'wb') as f:
            pickle.dump({
                'label2id': self._label2id,
                'id2label': self._id2label
            }, f)

    def _unload_additional_info(self):
        self._label2id, self._id2label = None, None

    def _load_additional_info(self):
        if os.path.exists(os.path.join(self._temp_cache, 'info.pkl')):
            with open(os.path.join(self._temp_cache, 'info.pkl'), 'rb') as f:
                loaded = pickle.load(f)
            self._label2id, self._id2label = loaded['label2id'], loaded['id2label']
        else:
            self._label2id, self._id2label = {}, {}

    @property
    def atype(self) -> str:
        return 'categorical'

    def _categorical_dimensions(self) -> List[Tuple[int, int]]:
        return [(0, 1), (1, self._calc_dim()+1)]

    def _calc_dim(self) -> int:
        return self._cat_cnt

    def _calc_fill_nan(self, original: pd.Series) -> str:
        original = original.astype(str)
        original.to_pickle(self._data_path)
        cat_cnt = len(self._label2id)
        categories = set(pd_mp.unique(original).dropna().reset_index(drop=True))
        for cat in categories:
            if cat not in self._label2id:
                self._label2id[cat], self._id2label[cat_cnt] = cat_cnt, cat
                cat_cnt += 1
        idx = 0
        while True:
            if f'nan_{idx}' not in self._label2id:
                label = f'nan_{idx}'
                return label
            idx += 1

    def _fit(self, original: pd.Series, nan_info: pd.DataFrame):
        transformed = self._transform(nan_info)
        self._transformed_columns = transformed.columns
        pd_to_pickle(transformed, self._transformed_path)
        self._cat_cnt = len(self._label2id)

    def _transform(self, nan_info: pd.DataFrame) -> pd.DataFrame:
        transformed = pd.DataFrame(columns=['is_nan'] + [f'cat_{i}' for i in self._id2label])
        transformed['is_nan'] = nan_info['is_nan']
        fast_map_dict(
            func=self._transform_row,
            dictionary=nan_info.to_dict(orient='index'),
            func_kwargs=dict(transformed=transformed)
        )
        return pd_mp.fillna(transformed, value=0).astype('float32')

    def _transform_row(self, i: int, row: Dict, transformed: pd.DataFrame):
        if not row['is_nan']:
            value = str(row['original'])
            if value in self._label2id:
                cat_id = self._label2id[value]
                transformed.loc[i, f'cat_{cat_id}'] = 1
            else:
                if True:#str(value) != '0':
                    data = pd.read_pickle(self._data_path)
                    print('???', value, [*self._label2id], data.unique())
                    _LOGGER.warning(f'Categorical value {value} is OOV.')
                    raise ValueError()

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        cat_ids = data.copy().set_axis([*range(data.shape[1])], axis=1).idxmax(axis=1)
        return cat_ids.apply(lambda x: self._id2label[x])

    def set_categories(self, labels: Collection):
        """
        **Args**:

        - `labels` (`Collection`): All categories of the categorical attribute to assign.
          It is the user's responsibility to check that the set is valid.
        """
        labels = [*labels]
        self._label2id = {l: i for i, l in enumerate(labels)}
        self._id2label = {i: l for i, l in enumerate(labels)}
        self._cat_cnt = len(labels)
        self._save_additional_info()


class CategoricalAttribute(BaseAttribute):
    """Attribute for categorical data."""
    def __init__(self, name: str, values: Optional[pd.Series] = None, temp_cache: str = '.temp'):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.
        """
        super().__init__(name, 'categorical', values.astype(str) if values is not None else values, temp_cache)

    def _create_transformer(self):
        self._transformer = CategoricalTransformer(self._temp_cache)

    def __copy__(self) -> "CategoricalAttribute":
        new_attr = super().__copy__()
        new_attr.__class__ = CategoricalAttribute
        return new_attr
