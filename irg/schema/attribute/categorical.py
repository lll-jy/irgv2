"""Handler for categorical data."""

from typing import Optional, List, Tuple

import pandas as pd

from .base import BaseAttribute, BaseTransformer


class CategoricalTransformer(BaseTransformer):
    """
    Transformer for categorical data.

    The transformed columns (after `is_nan`) are `cat_0`, `cat_1`, ..., `cat_k`
    for an attribute with k+1 categories.
    """
    def __init__(self, temp_cache: str = '.temp'):
        super().__init__(temp_cache)
        self._label2id, self._id2label = {}, {}

    @property
    def atype(self) -> str:
        return 'categorical'

    def _categorical_dimensions(self) -> List[Tuple[int, int]]:
        return [(0, 1), (1, self._calc_dim()+1)]

    def _calc_dim(self) -> int:
        return len(self._label2id)

    def _calc_fill_nan(self, original: pd.Series) -> str:
        original = original.astype(str)
        original.to_pickle(self._data_path)
        categories = set(original.dropna().reset_index(drop=True))
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

    def _fit(self, original: pd.Series, nan_info: pd.DataFrame):
        transformed = self._transform(nan_info)
        self._transformed_columns = transformed.columns
        transformed.to_pickle(self._transformed_path)

    def _transform(self, nan_info: pd.DataFrame) -> pd.DataFrame:
        transformed = pd.DataFrame()
        transformed['is_nan'] = nan_info['is_nan']
        for i in self._id2label:
            transformed[f'cat_{i}'] = 0
        for i, row in nan_info.iterrows():
            if not row['is_nan']:
                value = row['original']
                if value in self._label2id:
                    cat_id = self._label2id[row['original']]
                    transformed.loc[i, f'cat_{cat_id}'] = 1
                else:
                    # TODO: raise OOV warning
                    pass
        return transformed.astype('float32')

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        cat_ids = data.idxmax(axis=1)
        return cat_ids.apply(lambda x: self._id2label[x])


class CategoricalAttribute(BaseAttribute):
    """Attribute for categorical data."""
    def __init__(self, name: str, values: Optional[pd.Series] = None, temp_cache: str = '.temp'):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.
        """
        super().__init__(name, 'categorical', values, temp_cache)

    def _create_transformer(self):
        self._transformer = CategoricalTransformer(self._temp_cache)

    def __copy__(self) -> "CategoricalAttribute":
        new_attr = super().__copy__()
        new_attr.__class__ = CategoricalAttribute
        return new_attr
