from abc import ABC
from typing import Optional, Any, Union, Collection

import numpy as np
import pandas as pd
from torch import Tensor, from_numpy as tensor_from_numpy

from irg.utils.errors import NotFittedError


class BaseTransformer:
    def __init__(self):
        self._has_nan, self._fill_nan_val, self._nan_ratio = False, None, 0

        self._fitted = False
        self._original: Optional[pd.Series] = None
        self._transformed: Optional[pd.DataFrame] = None
        self._nan_info: Optional[pd.DataFrame] = None

    @property
    def atype(self) -> str:
        raise NotImplementedError()

    def fit(self, data: pd.Series, force_redo: bool = False):
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
        if self._fill_nan_val is None:
            self._fill_nan_val = self._calc_fill_nan(self._original)
        return self._fill_nan_val

    @property
    def hasnans(self) -> bool:
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

    def get_original_transformed(self, return_as: str = 'pandas') -> Optional[Union[pd.DataFrame, np.ndarray, Tensor]]:
        """
        Get the transformed dataframe built based on the data used for fitting.

        **Args**:

        - `return_as` (`str`): Valid values include
            * `'pandas'` for `pd.DataFrame`;
            * `'numpy'` for `np.ndarray`;
            * `'tensor'` for `torch.Tensor`.

        **Return**: The dataframe of transformed fitting data. The returned results is a copy.

        **Raise**:
        - `NotImplementedError` if the `return_as` is not recognized.
        - `NotFittedError` if the transformer is not yet fitted.
        """
        if not self._fitted:
            raise NotFittedError('Transformer', 'retrieving original transformed')
        return self._convert_data_as(self._transformed, return_as=return_as)

    @staticmethod
    def _convert_data_as(src: pd.DataFrame, return_as: str = 'pandas', copy: bool = True) -> \
            Optional[Union[pd.DataFrame, np.ndarray, Tensor]]:
        if return_as == 'pandas':
            if copy:
                return src.copy()
            else:
                return src
        if return_as == 'numpy':
            return src.to_numpy()
        if return_as == 'torch':
            return tensor_from_numpy(src.to_numpy())
        raise NotImplementedError(f'Unrecognized return type {return_as}. '
                                  f'Please choose from ["pandas", "numpy", and "torch"].')

    @staticmethod
    def _inverse_convert_data(src: Union[pd.DataFrame, np.ndarray, Tensor], columns: Optional[Collection]) \
            -> pd.DataFrame:
        if isinstance(src, pd.DataFrame):
            return src
        if isinstance(src, np.ndarray):
            return pd.DataFrame(src, columns=columns)
        if isinstance(src, Tensor):
            return pd.DataFrame(src.numpy(), columns=columns)
        raise NotImplementedError(f'Unrecognized return type {type(src)}. '
                                  f'Please make sure the input is one of [pd.DataFrame, np.ndarray, Tensor] type. ')

    def transform(self, data: pd.Series, return_as: str = 'pandas') -> pd.DataFrame:
        if not self._fitted:
            raise NotFittedError('Transformer', 'transforming other data')
        nan_info = self._construct_nan_info(data)
        transformed = self._transform(data, nan_info)
        return self._convert_data_as(transformed, return_as=return_as, copy=False)

    def _transform(self, data: pd.Series, nan_info: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError('Please specify attribute type for transformer.')

    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray, Tensor]) -> pd.Series:
        if not self._fitted:
            raise NotFittedError('Transformer', 'inversely transforming other data')
        data = self._inverse_convert_data(data, self._transformed.columns)
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
