"""Handler for ID attributes."""
from typing import Optional, Any

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


class SerialIDAttribute(BaseAttribute):
    """Attribute for serial ID data."""
    def __init__(self, name: str, values: Optional[pd.Series] = None):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        """
        super().__init__(name, 'serial_id', values)
        self._create_transformer()

    @property
    def atype(self) -> str:
        return 'serial_id'

    def _create_transformer(self):
        self._transformer = IdentityTransformer()

    def fit(self, values: pd.Series, force_redo: bool = False):
        raise TypeError('ID column cannot be fitted.')
