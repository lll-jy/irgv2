"""Handler for datetime-related data."""
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .base import BaseAttribute
from .numerical import NumericalTransformer


class DatetimeTransformer(NumericalTransformer):
    """Transformer for datetime data."""
    def __init__(self, date_format: str = '%Y-%m-%d', **kwargs):
        """
        Transformer for datetime data.

        **Args**:

        - `date_format` (`str`) [default `'%Y-%m-%d'`]: Datetime format string that is processable by
          [`Python`'s in-built `datetime.strftime`](https://docs.python.org/3/library/datetime.html).
          This is the format for how the datetime data is and will be represented.
        - `kwargs`: Arguments for [`NumericalTransformer`](numerical#NumericalTransformer),
          for the data after applying `toordinal`. Thus, rounding will be overridden as 0.
        """
        super().__init__(**kwargs)
        self._format = date_format
        self._rounding = 0
        self._original_as_date: Optional[pd.Series] = None

    @property
    def atype(self) -> str:
        return 'datetime'

    def _calc_fill_nan(self) -> datetime:
        val = self._original.mean()
        if pd.isnull(val):
            return datetime.now()
        return val

    def _fit(self):
        self._original_as_date = self._original
        self._original = self._original.apply(lambda x: x.toordinal())
        super()._fit()

    def _transform(self, nan_info: pd.DataFrame) -> pd.DataFrame:
        nan_info = nan_info.copy()
        nan_info['original'] = nan_info['original'].apply(lambda x: x.toordinal())
        return super()._transform(nan_info)

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        numerical_result = super()._inverse_transform(data)
        datetime_result = numerical_result.astype(int).apply(datetime.fromordinal)
        return datetime_result.apply(lambda x: datetime.strftime(x, self._format)).astype('datetime64')


class DatetimeAttribute(BaseAttribute):
    """Attribute for datetime data."""
    def __init__(self, name: str, values: Optional[pd.Series] = None, **kwargs):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        - `kwargs`: Arguments for `DatetimeTransformer`.
        """
        super().__init__(name, 'numerical', values)
        self._kwargs = kwargs

    @property
    def atype(self) -> str:
        return 'datetime'

    def _create_transformer(self):
        self._transformer = DatetimeTransformer(**self._kwargs)
