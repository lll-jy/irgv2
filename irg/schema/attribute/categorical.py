"""Handler for categorical data."""

from typing import Any

import pandas as pd

from .base import BaseAttribute, BaseTransformer


class CategoricalTransformer(BaseTransformer):
    """Transformer for categorical data."""
    def __init__(self):
        super().__init__()
        self._label2id, self._id2label = {}, {}

    @property
    def atype(self) -> str:
        return 'categorical'

    def _calc_dim(self) -> int:
        return len(self._label2id)

    def _calc_fill_nan(self) -> Any:
        categories = set(self._original.dropna().astype('str').reset_index(drop=True))
        cat_cnt = 0
        for cat in categories:
            self._label2id[cat], self._id2label[cat_cnt] = cat_cnt, cat
            cat_cnt += 1
        idx = 0
        while True:
            if f'nan_{idx}' not in self._label2id:
                label = f'nan_{idx}'
                return label
            idx += 1

    def _fit(self):
        self._transformed = self._transform(self._nan_info)

    def _transform(self, nan_info: pd.DataFrame) -> pd.DataFrame:
        transformed = pd.DataFrame()
        if self._has_nan:
            transformed['is_nan'] = nan_info['is_nan']
        for i in self._id2label:
            transformed[f'cat_{i}'] = 0
        for i, row in nan_info.iterrows():
            if not row['is_nan']:
                cat_id = self._label2id[row['original']]
                transformed.loc[i, f'cat_{cat_id}'] = 1
        return transformed

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        if self._has_nan:
            cat_data = data.drop(columns=['is_nan'])
        else:
            cat_data = data
        cat_ids = cat_data.idxmax(axis=1)
        return cat_ids.apply(lambda x: self._id2label[x])


class CategoricalAttribute(BaseAttribute):
    """Attribute for categorical data."""
    @property
    def atype(self) -> str:
        return 'categorical'

    def _create_transformer(self):
        self._transformer = CategoricalTransformer()
