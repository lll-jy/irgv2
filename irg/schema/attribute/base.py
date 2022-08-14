"""Abstract base class of attributes."""

from abc import ABC, abstractmethod
from typing import Optional, Any, Collection, List, Tuple
import os

import numpy as np
import pandas as pd

from ...utils.errors import NotFittedError
from ...utils.misc import convert_data_as, inverse_convert_data, Data2D


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

    In particular, the output of transform and input for inverse transform must have first column 'is_nan'.
    The remaining columns depends on actual data type.
    """
    def __init__(self, temp_cache: str = '.temp'):
        """
        **Args**:

        - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.
        """
        self._has_nan, self._fill_nan_val, self._temp_cache = False, None, temp_cache

        self._fitted, self._dim = False, -1
        self._transformed: Optional[pd.DataFrame] = None

    @property
    @abstractmethod
    def atype(self) -> str:
        """
        Transformer's corresponding attribute type.
        """
        raise NotImplementedError()

    @property
    def dim(self) -> int:
        """
        Number of dimensions of transformed data.
        """
        if not self._fitted:
            raise NotFittedError('Transformer', 'retrieving the dimension')
        if self._dim < 0:
            self._dim = self._calc_dim()
        nan_dim = 1 if self.has_nan else 0
        return self._dim + nan_dim

    @property
    def transformed_columns(self) -> Collection[str]:
        """
        Transformed column names.
        """
        if not self._fitted:
            raise NotFittedError('Transformer', 'getting transformed column names')
        return self._transformed.columns

    @abstractmethod
    def _calc_dim(self) -> int:
        raise NotImplementedError()

    @property
    def _data_path(self) -> str:
        return os.path.join(self._temp_cache, 'data.pkl')

    @property
    def _nan_info_path(self) -> str:
        return os.path.join(self._temp_cache, 'nan_info.pkl')

    def fit(self, data: pd.Series, force_redo: bool = False):
        """
        Fit the attribute's normalization transformers.

        **Args**:

        - `values` (`pd.Series`): The values of the attribute to fit the normalization transformers.
          Typically this is the data in the real database's table's attribute.
        - `force_redo` (`bool`) [default `False`]: Whether to re-fit if the attribute is already fitted.
          Default is `False`.
        """
        if self._fitted and not force_redo:
            return
        data.to_pickle(self._data_path)
        self._fit_for_nan(data)
        self._fit(data)
        self._fitted = True

    def _fit_for_nan(self, original: pd.Series):
        self._has_nan = original.hasnans
        self._construct_nan_info(original).to_pickle(self._nan_info_path)

    @property
    def fill_nan_val(self) -> Any:
        """
        The value used for filling `NaN`'s.
        """
        return self._fill_nan_val

    @property
    def has_nan(self) -> bool:
        """
        Whether the attribute contains `NaN` values.
        """
        if not self._fitted:
            raise NotFittedError('Transformer', 'checking whether the attribute has NaN')
        return self._has_nan

    @abstractmethod
    def _calc_fill_nan(self, original: pd.Series) -> Any:
        raise NotImplementedError('Fill NaN value is not implemented for base transformer.')

    def _construct_nan_info(self, original: pd.Series) -> pd.DataFrame:
        nan_info = pd.DataFrame()
        nan_info['original'] = original
        self._fill_nan_val = self._calc_fill_nan(original)
        nan_info['is_nan'] = original.isnull()
        nan_info['original'].fillna(self._fill_nan_val, inplace=True)
        return nan_info

    @abstractmethod
    def _fit(self, original: pd.Series, nan_info: pd.DataFrame):
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
        transformed = self._transform(nan_info)
        return convert_data_as(transformed, return_as=return_as, copy=False)

    @abstractmethod
    def _transform(self, nan_info: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError('Please specify attribute type for transformer.')

    def inverse_transform(self, data: Data2D, nan_ratio: Optional[float] = None, nan_thres: Optional[float] = None) \
            -> pd.Series:
        """
        Inversely transform a normalized dataframe back to the original raw data form.

        **Args**:

        - `values` (`Data2D`): The normalized data.
        - `nan_ratio` (`Optional[float]`): Ratio of the data to be NaN.
          Under default setting, it will follow the ratio of the real data used for fitting the attribute.
        - `nan_thres` (`Optional[float]`): Threshold to tell that a value is `NaN`.
          The 'is_nan' prediction is true if the value is larger than the threshold.
          Only one of `nan_ratio` and `nan_thres` can be specified.

        **Return**: The recovered series of raw data.

        **Raise**: `NotFittedError` if the transformer is not yet fitted.
        """
        if not self._fitted:
            raise NotFittedError('Transformer', 'inversely transforming other data')
        data = inverse_convert_data(data, self._transformed.columns)
        core_data = data.drop(columns=['is_nan']) if self._has_nan else data
        recovered_no_nan = self._inverse_transform(core_data)
        if nan_thres is not None:
            threshold = nan_thres
        else:
            if nan_ratio is None:
                original = pd.read_pickle(self._data_path)
                nan_ratio = original.count() / len(original)
            threshold = data['is_nan'].quantile(1 - nan_ratio)
        recovered_no_nan[data['is_nan'] > threshold] = np.nan
        return recovered_no_nan

    @abstractmethod
    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError('Please specify attribute type for transformer.')

    def categorical_dimensions(self, base: int = 0) -> List[Tuple[int, int]]:
        """
        Get dimension indices normalized as if categories.

        **Args**:

        - `base` (`int`): The base index. Default is 0.

        **Return**: List of [L, R) pairs denoting the ranges representing categorical information.
        """
        if not self._fitted:
            raise NotFittedError('Transformer', 'getting categorical columns')
        dimensions = self._categorical_dimensions()
        return [(l+base, r+base) for l, r in dimensions]

    @abstractmethod
    def _categorical_dimensions(self) -> List[Tuple[int, int]]:
        raise NotImplementedError('Please specify attribute type for transformer.')


class BaseAttribute(ABC):
    """
    Abstract base class for attributes.
    """
    def __init__(self, name: str, attr_type: str, values: Optional[pd.Series] = None):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `attr_type` (`str`): Attribute type name.
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        """
        self._name, self._attr_type = name, attr_type

        self._transformer: Optional[BaseTransformer] = None
        if values is not None and attr_type != 'id':
            self.fit(values)

    def rename(self, new_name: str, inplace: bool = True) -> Optional["BaseAttribute"]:
        """
        Rename the current attribute.

        **Args**:

        - `new_name` (`str`): New name of the attribute.
        - `inplace` (`bool`): Whether to change name in-place. Default is `True`.
          If set `True`, nothing is returned.

        **Return**: Renamed new attribute if inplace is `False`.
        """
        if inplace:
            self._name = new_name
        else:
            new_attr = self.__copy__()
            new_attr._name = new_name
            return new_attr

    def __copy__(self) -> "BaseAttribute":
        new_attr = BaseAttribute(self._name, self._attr_type)
        new_attr._transformer = self._transformer
        return new_attr

    @property
    def atype(self) -> str:
        """
        The attribute type.
        """
        return self._attr_type

    @property
    def name(self) -> str:
        """
        The name of the attribute.
        """
        return self._name

    @property
    def fill_nan_val(self) -> Any:
        """
        The value used for filling `NaN`'s.
        """
        return self._transformer.fill_nan_val

    @property
    def transformed_columns(self) -> Collection[str]:
        """
        Transformed column names.
        """
        return self._transformer.transformed_columns

    def categorical_dimensions(self, base: int = 0) -> List[Tuple[int, int]]:
        """
        Get dimension indices normalized as if categories.

        **Args**:

        - `base` (`int`): The base index. Default is 0.

        **Return**: List of [L, R) pairs denoting the ranges representing categorical information.
        """
        self._transformer.categorical_dimensions(base)

    def fit(self, values: pd.Series, force_redo: bool = False):
        """
        Fit the attribute's normalization transformers.

        **Args**:

        - `values` (`pd.Series`): The values of the attribute to fit the normalization transformers.
          Typically this is the data in the real database's table's attribute.
        - `force_redo` (`bool`) [default `False`]: Whether to re-fit if the attribute is already fitted.
          Default is `False`.
        """
        self._create_transformer()
        self._transformer.fit(values, force_redo)

    @abstractmethod
    def _create_transformer(self):
        raise NotImplementedError('Please specify attribute type for preparing corresponding transformer.')

    def get_original_transformed(self, return_as: str = 'pandas') -> Data2D:
        """
        Get the transformed dataframe built based on the data used for fitting.

        **Args**:

        - `return_as` (`str`): [Valid types to convert](../../utils/misc#convert_data_as).

        **Return**: The data of transformed fitting data in the desired format. The returned results is a copy.

        **Raise**: `NotFittedError` if the transformer is not yet fitted.
        """
        return self._transformer.get_original_transformed(return_as)

    def transform(self, data: pd.Series, return_as: str = 'pandas') -> pd.DataFrame:
        """
        Transform a new set of data for this attribute based on the fitted result.

        **Args**:

        - `values` (`pd.Series`): The values to be transformed.
        - `return_as` (`str`): [Valid types to return](../../utils/misc#convert_data_as).

        **Return**: The data of transformed fitting data in the desired format.

        **Raise**: `NotFittedError` if the transformer is not yet fitted.
        """
        return self._transformer.transform(data, return_as)

    def inverse_transform(self, data: Data2D) -> pd.Series:
        """
        Inversely transform a normalized dataframe back to the original raw data form.

        **Args**:

        - `values` (`Data2D`): The normalized data.

        **Return**: The recovered series of raw data.

        **Raise**: `NotFittedError` if the transformer is not yet fitted.
        """
        return self._transformer.inverse_transform(data)
