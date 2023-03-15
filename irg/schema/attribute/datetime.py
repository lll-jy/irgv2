"""Handler for datetime-related data."""
from datetime import datetime, timedelta
import numbers
from typing import Optional
import os

import numpy as np
import pandas as pd

from .base import BaseAttribute
from .numerical import NumericalTransformer


class DatetimeTransformer(NumericalTransformer):
    """Transformer for datetime data."""
    def __init__(self, date_format: str = '%Y-%m-%d', **kwargs):
        """
        **Args**:
        - `date_format` (`str`) [default `'%Y-%m-%d'`]: Datetime format string that is processable by
          [`Python`'s in-built `datetime.strftime`](https://docs.python.org/3/library/datetime.html).
          This is the format for how the datetime data is and will be represented.
        - `kwargs`: Arguments for [`NumericalTransformer`](numerical#NumericalTransformer),
          for the data by the number of seconds to mean. Thus, rounding will be overridden as 0.
        """
        super().__init__(**kwargs)
        self._format = date_format
        self._mean = None

    def _datetime_to_number(self, dt: Optional[datetime]) -> float:
        if pd.isnull(dt) or dt is None:
            return np.nan
        if hasattr(self, '_mean') and self._mean is not None:
            return (dt - self._mean).total_seconds()
        else:
            return dt.toordinal()

    def _number_to_datetime(self, number: Optional[float]) -> datetime:
        if pd.isnull(number) or number is None:
            return np.nan
        if hasattr(self, '_mean') and self._mean is not None:
            return self._mean + timedelta(seconds=number)
        else:
            return datetime.fromordinal(int(number))

    @property
    def _as_date_path(self) -> str:
        return os.path.join(self._temp_cache, 'date.pkl')

    @property
    def atype(self) -> str:
        return 'datetime'

    def _calc_fill_nan(self, original: pd.Series) -> datetime:
        original = original.astype('datetime64[ns]')
        val = original.mean()
        if pd.isnull(val):
            return datetime.now()
        return val

    def _fit(self, original: pd.Series, nan_info: pd.DataFrame):
        self._mean = original.mean()
        index = nan_info.index
        original = original.astype('datetime64[ns]')
        original.to_pickle(self._as_date_path)
        original = original.apply(self._datetime_to_number).astype('float32')
        original.to_pickle(self._data_path)
        nan_info = nan_info.reset_index(drop=True)
        nan_info.loc[:, 'original'] = nan_info['original'].astype('datetime64[ns]')\
            .apply(self._datetime_to_number).astype('float32')
        nan_info = nan_info.astype({'original': 'float32'})
        if original.std() < 10:
            raise ValueError('wrong!!!', original.std())
        nan_info.index = index
        super()._fit(original, nan_info)

    def _transform(self, nan_info: pd.DataFrame) -> pd.DataFrame:
        cache = self._temp_cache
        while '/' in cache:
            cache = cache[cache.index('/')+1:]

        index = nan_info.index
        nan_info = nan_info.reset_index(drop=False)
        if len(nan_info) > 0:
            nan_info['original'] = nan_info['original'].apply(
                lambda x: x if pd.isnull(x) or not isinstance(x, datetime) else self._datetime_to_number(x)
            ).astype('float32')
        else:
            nan_info.loc[:, 'original'] = 0
            nan_info['original'] = nan_info['original'].astype('float32')
        nan_info.index = index
        transformed = super()._transform(nan_info)
        return transformed

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        numerical_result = super()._inverse_transform(data)
        datetime_result = numerical_result.apply(self._number_to_datetime)
        formatted = datetime_result.apply(lambda x: x if pd.isnull(x) else x.strftime(self._format))
        formatted = formatted.astype('datetime64[ns]')
        original = pd.read_pickle(self._as_date_path)
        original = original.apply(self._datetime_to_number).astype('float32')
        return formatted


class DatetimeAttribute(BaseAttribute):
    """Attribute for datetime data."""
    def __init__(self, name: str, values: Optional[pd.Series] = None, temp_cache: str = '.temp', **kwargs):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.
        - `kwargs`: Arguments for `DatetimeTransformer`.
        """
        self._kwargs = kwargs
        super().__init__(name, 'datetime', values, temp_cache)

    def _create_transformer(self):
        self._transformer = DatetimeTransformer(temp_cache=self._temp_cache, **self._kwargs)

    def __copy__(self) -> "DatetimeAttribute":
        new_attr = super().__copy__()
        new_attr.__class__ = DatetimeAttribute
        new_attr._kwargs = self._kwargs
        return new_attr


class TimedeltaTransformer(NumericalTransformer):
    """Transformer for timedelta data."""
    def __init__(self, delta_format: str = '%H:%M:%S', **kwargs):
        """
        **Args**:

        - `delta_format` (`str`) [default `'%H:%M:%S'`]: Datetime format string that is processable by
          [`Python`'s in-built `datetime.strftime`](https://docs.python.org/3/library/datetime.html).
          This is the format for how the datetime data is and will be represented.
        - `kwargs`: Arguments for [`NumericalTransformer`](numerical#NumericalTransformer),
          for the data after applying `total_seconds`.
        """
        super().__init__(**kwargs)
        self._format = delta_format
        self._original_as_delta: Optional[pd.Series] = None

    @property
    def _as_delta_path(self) -> str:
        return os.path.join(self._temp_cache, 'delta.pkl')

    @property
    def atype(self) -> str:
        return 'timedelta'

    def _calc_fill_nan(self, original: pd.Series) -> timedelta:
        val = original.mean()
        if pd.isnull(val):
            return timedelta(seconds=0)
        return val

    def _fit(self, original: pd.Series, nan_info: pd.DataFrame):
        index = nan_info.index
        original.to_pickle(self._as_delta_path)
        original = original.apply(lambda x: np.nan if pd.isnull(x) else
        x.total_seconds() if isinstance(x, timedelta) else x).astype('float32')
        original.to_pickle(self._data_path)
        nan_info = nan_info.reset_index(drop=True)
        original.index = nan_info.index
        nan_info.loc[:, 'original'] = original
        nan_info.index = index
        original.index = index
        super()._fit(original, nan_info)

    def _transform(self, nan_info: pd.DataFrame) -> pd.DataFrame:
        index = nan_info.index
        nan_info = nan_info.reset_index(drop=True).copy()
        nan_info['original'] = nan_info['original'].apply(
            lambda x: x if pd.isnull(x) or isinstance(x, numbers.Number) else x.total_seconds()
        ).astype('float32')
        nan_info.index = index
        return super()._transform(nan_info)

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        numerical_result = super()._inverse_transform(data)
        timedelta_result = numerical_result.apply(lambda x: x if pd.isnull(x) else timedelta(seconds=x))
        return timedelta_result.apply(
            lambda x: x if pd.isnull(x) else (datetime.strptime('', '') + x).strftime(self._format)
        )


class TimedeltaAttribute(BaseAttribute):
    """Attribute for timedelta data."""
    def __init__(self, name: str, values: Optional[pd.Series] = None, temp_cache: str = '.temp', **kwargs):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.
        - `kwargs`: Arguments for `TimedeltaTransformer`.
        """
        self._kwargs = kwargs
        super().__init__(name, 'timedelta', values, temp_cache)

    def _create_transformer(self):
        self._transformer = TimedeltaTransformer(temp_cache=self._temp_cache, **self._kwargs)

    def __copy__(self) -> "TimedeltaAttribute":
        new_attr = super().__copy__()
        new_attr.__class__ = TimedeltaAttribute
        new_attr._kwargs = self._kwargs
        return new_attr
