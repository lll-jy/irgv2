"""Abstract base class of attributes."""

from abc import ABC
from typing import Optional, Any

import numpy as np
import pandas as pd

from irg.utils.errors import NotFittedError
from irg.utils.misc import convert_data_as, inverse_convert_data, Data2D


class BaseTransformer:
    """
    Abstract base class for transformer.

    It is assumed that there is no index errors.
    Namely, all input data, especially in terms of non-consecutive-index-tolerant type like `pd.DataFrame` and
    `pd.Series`, will still always have indices [0..n] where n is the length of the data.
    Also, all input 2D data, including those with header names, must have columns aligned.
    That is, the order of the columns must not be messed up.
    For example, a table with columns ['A', 'B'] cannot be inputted as ['B', 'A'].
    The above assumptions will not be reported in any form if violated.
    So please make pre-processing and post-processing steps if necessary.

    In particular, the output of transform and input for inverse transform must have first column 'is_nan'
    if the data can be nan (i.e. `self.has_nan` is `True`).
    The remaining columns depends on actual data type.
    """
    def __init__(self):
        self._has_nan, self._fill_nan_val, self._nan_ratio = False, None, 0

        self._fitted = False
        self._original: Optional[pd.Series] = None
        self._transformed: Optional[pd.DataFrame] = None
        self._nan_info: Optional[pd.DataFrame] = None

    @property
    def atype(self) -> str:
        """
        Transformer's corresponding attribute type.
        """
        raise NotImplementedError()

    def fit(self, data: pd.Series, force_redo: bool = False):
        """
        Fit the attribute's normalization transformers.

        **Args**:

        - `values` (`pd.Series`): The values of the attribute to fit the normalization transformers.
          Typically this is the data in the real database's table's attribute.
        - `force_redo` (`bool`): Whether to re-fit if the attribute is already fitted.
          Default is `False`.
        """
        if self._fitted and not force_redo:
            return
        self._original = data
        self._fit_for_nan()
        self._fit()
        self._fitted = True

    def _fit_for_nan(self):
        self._has_nan = self._original.hasnans
        self._nan_info = self._construct_nan_info(self._original)
        if self._has_nan:
            self._nan_ratio = 1 - self._original.count() / len(self._original)

    @property
    def fill_nan_val(self) -> Any:
        """
        The value used for filling `NaN`'s.
        """
        if self._fill_nan_val is None:
            self._fill_nan_val = self._calc_fill_nan(self._original)
        return self._fill_nan_val

    @property
    def has_nan(self) -> bool:
        """
        Whether the attribute contains `NaN` values.
        """
        return self._has_nan

    def _calc_fill_nan(self, data: pd.Series):
        raise NotImplementedError('Fill NaN value is not implemented for base transformer.')

    def _construct_nan_info(self, original: pd.Series):
        nan_info = pd.DataFrame()
        nan_info['original'] = original
        if self._has_nan:
            nan_info['is_nan'] = original.isnull()
            nan_info['original'].fillna(self.fill_nan_val, inplace=True)
        return nan_info

    def _fit(self):
        raise NotImplementedError('Fit is not implemented for base transformer.')

    def get_original_transformed(self, return_as: str = 'pandas') -> Data2D:
        """
        Get the transformed dataframe built based on the data used for fitting.

        **Args**:

        - `return_as` (`str`): [Valid types to convert](../../utils/misc#convert_data_as).

        **Return**: The data of transformed fitting data in the desired format. The returned results is a copy.

        **Raise**: `NotFittedError` if the transformer is not yet fitted.
        """
        if not self._fitted:
            raise NotFittedError('Transformer', 'retrieving original transformed')
        return convert_data_as(self._transformed, return_as=return_as)

    def transform(self, data: pd.Series, return_as: str = 'pandas') -> pd.DataFrame:
        """
        Transform a new set of data for this attribute based on the fitted result.

        **Args**:

        - `values` (`pd.Series`): The values to be transformed.
        - `return_as` (`str`): [Valid types to return](../../utils/misc#convert_data_as).

        **Return**: The data of transformed fitting data in the desired format.

        **Raise**: `NotFittedError` if the transformer is not yet fitted.
        """
        if not self._fitted:
            raise NotFittedError('Transformer', 'transforming other data')
        nan_info = self._construct_nan_info(data)
        transformed = self._transform(data, nan_info)
        return convert_data_as(transformed, return_as=return_as, copy=False)

    def _transform(self, data: pd.Series, nan_info: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError('Please specify attribute type for transformer.')

    def inverse_transform(self, data: Data2D) -> pd.Series:
        """
        Inversely transform a normalized dataframe back to the original raw data form.

        **Args**:

        - `values` (`Data2D`): The normalized data.

        **Return**: The recovered series of raw data.

        **Raise**: `NotFittedError` if the transformer is not yet fitted.
        """
        if not self._fitted:
            raise NotFittedError('Transformer', 'inversely transforming other data')
        data = inverse_convert_data(data, self._transformed.columns)
        recovered_no_nan = self._inverse_transform(data)
        if not self._has_nan:
            return recovered_no_nan
        threshold = data['is_nan'].quantile(1 - self._nan_ratio)
        recovered_no_nan[data['is_nan'] > threshold] = np.nan
        return recovered_no_nan

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError('Please specify attribute type for transformer.')


class BaseAttribute(ABC):
    def __init__(self, name: str, attr_type: str, values: Optional[pd.Series] = None):
        self._name, self._attr_type = name, attr_type

        self._original_values = values
