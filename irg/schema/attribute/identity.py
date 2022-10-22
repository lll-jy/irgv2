"""Handler for ID attributes."""
from typing import Optional, List, Tuple, Collection

import pandas as pd

from .base import BaseAttribute, BaseTransformer
from ...utils.misc import Data2D
from ...utils.io import pd_to_pickle


class IdentityTransformer(BaseTransformer):
    """Transformer that retain values identical as originally given."""

    def _unload_additional_info(self):
        pass

    def _load_additional_info(self):
        pass

    def _save_additional_info(self):
        pass

    @property
    def atype(self) -> str:
        return 'identity'

    def _calc_dim(self) -> int:
        return 1

    def _calc_fill_nan(self, original: pd.Series) -> str:
        return 'nan'

    def _fit(self, original: pd.Series, nan_info: pd.DataFrame):
        transformed = self._transform(nan_info)
        self._transformed_columns = transformed.columns
        pd_to_pickle(transformed, self._transformed_path)
        self._cat_cnt = 1

    def _transform(self, nan_info: pd.DataFrame) -> pd.DataFrame:
        return nan_info[['is_nan']]

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        return data

    def inverse_transform(self, data: Data2D, nan_ratio: Optional[float] = None, nan_thres: Optional[float] = None) \
            -> pd.Series:
        print('in for id inverse', data.columns)
        nan_indicator, _ = self._inverse_nan_info(data, nan_ratio, nan_thres)
        return nan_indicator

    def _categorical_dimensions(self) -> List[Tuple[int, int]]:
        return [(0, 1)]


class SerialIDAttribute(BaseAttribute):
    """Attribute for serial ID data."""
    def __init__(self, name: str, values: Optional[pd.Series] = None, temp_cache: str = '.temp',
                 generator: str = 'lambda x: x'):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.
        - `generator` (`str`): An executable string by `eval` function that returns a function mapping every
          non-negative integer to a unique ID.
        """
        super().__init__(name, 'id', values, temp_cache)
        self._create_transformer()
        self._generator = generator

    def categorical_dimensions(self, base: int = 0) -> List[Tuple[int, int]]:
        return [(base, base+1)]

    def __copy__(self) -> "SerialIDAttribute":
        new_attr = super().__copy__()
        new_attr.__class__ = SerialIDAttribute
        new_attr._generator = self._generator
        return new_attr

    def _create_transformer(self):
        self._transformer = IdentityTransformer(self._temp_cache)

    def inverse_transform(self, data: Data2D) -> pd.Series:
        nan_res = self._transformer.inverse_transform(data)
        print('result from transformer ID', nan_res.head())
        no_nan = self.generate(len(data))
        print('generated is', no_nan.head())
        return no_nan[nan_res]

    def generate(self, n: int) -> pd.Series:
        """
        Generate data for this ID attribute.

        **Args**:

        - `n` (`int`): The number of instances to be generated.

        **Return**: A `pd.Series` containing the generated IDs, by applying the generator function to 0 to n-1.
        """
        return pd.Series([i for i in range(n)]).apply(eval(self._generator))


class RawTransformer(IdentityTransformer):
    """
    Transformer that retain original raw value, and fill NaN with 0. Typically used for float columns
    within adequate range.
    """
    @property
    def atype(self) -> str:
        return 'raw'

    def _calc_fill_nan(self, original: pd.Series) -> float:
        return 0

    def categorical_dimensions(self, base: int = 0) -> List[Tuple[int, int]]:
        return [(base, base+1)]

    @property
    def transformed_columns(self) -> Collection[str]:
        return ['data']


class RawAttribute(BaseAttribute):
    """Attribute that does not need any transformation normalization (other than potential NaN due to joining)."""
    def __init__(self, name: str, values: Optional[pd.Series] = None, temp_cache: str = '.temp'):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.
        """
        super().__init__(name, 'raw', values, temp_cache)
        self._create_transformer()

    def _create_transformer(self):
        self._transformer = RawTransformer(self._temp_cache)

    def __copy__(self) -> "RawAttribute":
        new_attr = super().__copy__()
        new_attr.__class__ = RawAttribute
        return new_attr
