"""Handler for ID attributes."""
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

from .base import BaseAttribute, BaseTransformer


class IdentityTransformer(BaseTransformer):
    @property
    def atype(self) -> str:
        return 'identity'

    def _calc_dim(self) -> int:
        return 1

    def _calc_fill_nan(self) -> float:
        return np.nan

    def _fit(self):
        pass

    def _transform(self, nan_info: pd.DataFrame) -> pd.DataFrame:
        return nan_info[['original']]

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        return data['original']

    def _categorical_dimensions(self) -> List[Tuple[int, int]]:
        return [(0, 1)]


class SerialIDAttribute(BaseAttribute):
    """Attribute for serial ID data."""
    def __init__(self, name: str, values: Optional[pd.Series] = None, generator: str = 'lambda x: x'):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        - `generator` (`str`): An executable string by `eval` function that returns a function mapping every
          non-negative integer to a unique ID.
        """
        super().__init__(name, 'id', values)
        self._create_transformer()
        self._generator = generator

    def _create_transformer(self):
        self._transformer = IdentityTransformer()

    def fit(self, values: pd.Series, force_redo: bool = False):
        raise TypeError('ID column cannot be fitted.')

    def generate(self, n: int) -> pd.Series:
        """
        Generate data for this ID attribute.

        **Args**:

        - `n` (`int`): The number of instances to be generated.

        **Return**: A `pd.Series` containing the generated IDs, by applying the generator function to 0 to n-1.
        """
        return pd.Series([i for i in range(n)]).apply(eval(self._generator))

