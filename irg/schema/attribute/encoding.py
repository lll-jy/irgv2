"""Handler for encoding data."""
from typing import Optional, Dict, List, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from .base import BaseAttribute, BaseTransformer
from ...utils.misc import load_from


class EncodingTransformer(BaseTransformer):
    """Transformer for encoding data, where each value associates with a vector representation."""
    def __init__(self):
        super().__init__()
        self._vocab: Optional[Dict[str, List[Union[int, float]]]] = None
        self._vocab_dim = -1
        self._knn = KNeighborsClassifier(n_neighbors=1)
        self._mean_enc: Optional[List[float]] = None

    @property
    def atype(self) -> str:
        return 'encoding'

    def load_vocab(self, check_dim: bool = True, **kwargs):
        """
        Load vocabulary file.

        **Args**:

        - `check_dim` (`bool`) [default `True`]: Whether to check the validity of loaded data format.
          It must be able to be interpreted as a `dict` from `str` to a vector of numbers.
          And the length of the vectors should be the same for all words.
          Also, the vocab cannot be empty.
          '[UNK]' is reserved for recognizing unseen values. It will not be checked because input with this value
          is likely to express the same meaning.
        - `kwargs`: Arguments for [load_from](../utils/misc#load_from).
        """
        self._vocab = load_from(**kwargs)
        if check_dim:
            self._check_vocab()

    def _check_vocab(self):
        if not isinstance(self._vocab, Dict):
            raise ValueError(f'The vocabulary must be a dict. Got {type(self._vocab)}.')
        dim = -1
        for k, v in self._vocab.items():
            if dim < 0:
                dim = len(v)
            else:
                if dim != len(v):
                    raise ValueError(f'Encoding dimension does not match. Want {dim}, got {len(v)}.')
        if dim < 0:
            raise ValueError('The vocabulary must not be empty.')

    def _calc_dim(self) -> int:
        return self._vocab_dim

    def _calc_fill_nan(self) -> str:
        return '[UNK]'

    def _fit(self):
        values = [*self._vocab.values()]
        self._mean_enc = np.array(values).mean(axis=0)
        self._vocab_dim = len(values[0])
        self._knn.fit(values, [*self._vocab.keys()])
        self._transformed = self._transform(self._nan_info)

    def _transform(self, nan_info: pd.DataFrame) -> pd.DataFrame:
        nan_info['original'] = nan_info['original'].astype(str)
        col_names = [f'enc_{i}' for i in range(self._vocab_dim)]
        col_names = ['is_nan'] + col_names
        transformed = pd.DataFrame(columns=col_names)
        if self._has_nan:
            transformed['is_nan'] = nan_info['is_nan']
        for i, row in nan_info.iterrows():
            if row['is_nan']:
                transformed.iloc[i, 1:] = 0
            else:
                transformed.iloc[i, 1:] = self._vocab.get(row['original'], self._mean_enc)
        return transformed

    def _categorical_dimensions(self) -> List[Tuple[int, int]]:
        return [(0, 1)]

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(self._knn.predict(data))


class EncodingAttribute(BaseAttribute):
    """Attribute for encoding data, where each value associates with a vector representation."""

    def __init__(self, name: str, vocab_file: str, engine: Optional[str] = None, values: Optional[pd.Series] = None):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `vocab_file` (`str`): File to vocabulary.
        - `engine` (`Optional[str]`): Engine for [load_from](../utils/misc#load_from).
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        """
        super().__init__(name, 'encoding', values)
        self._vocab_file, self._engine = vocab_file, engine

    def _create_transformer(self):
        self._transformer = EncodingTransformer()
        self._transformer.load_vocab(file_path=self._vocab_file, engine=self._engine)
