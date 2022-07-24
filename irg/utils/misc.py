from typing import Optional, Union, Collection

import numpy as np
import pandas as pd
from torch import Tensor, from_numpy as tensor_from_numpy


__all__ = (
    'Data2D',
    'convert_data_as',
    'inverse_convert_data'
)


Data2D = Union[pd.DataFrame, np.ndarray, Tensor]
"""2D data type, including `pd.DataFrame`, `np.ndarray`, and `torch.Tensor`."""


def convert_data_as(src: pd.DataFrame, return_as: str = 'pandas', copy: bool = True) -> Data2D:
    """
    Convert a pd.DataFrame to desired data type.

    **Args**:

    - `src` (`pd.DataFrame`): The data to be converted.
    - `return_as` (`str`) [default 'pandas']: Valid values include
        * `'pandas'` for `pd.DataFrame`;
        * `'numpy'` for `np.ndarray`;
        * `'tensor'` for `torch.Tensor`.
    - `copy` (`bool`) [default `True`]: Whether to make a copy when returning the data.

    **Return**: The converted data of the desired type.

    **Raise**: `NotImplementedError` if the `return_as` is not recognized.
    """
    if return_as == 'pandas':
        if copy:
            return src.copy()
        else:
            return src
    if return_as == 'numpy':
        return src.to_numpy(copy=copy)
    if return_as == 'torch':
        return tensor_from_numpy(src.to_numpy())
    raise NotImplementedError(f'Unrecognized return type {return_as}. '
                              f'Please choose from ["pandas", "numpy", and "torch"].')


def inverse_convert_data(src: Data2D, columns: Optional[Collection]) -> pd.DataFrame:
    """
    Convert a valid `Data2D` type to `pd.DataFrame`.

    **Args**:

    - `src` (`Data2D`): The data to be converted.
    - `columns` (`Optional[Collection]`): The column names of the dataframe.

    **Return**: The converted dataframe.

    **Raise**: `NotImplementedError` if the input `src` type is not recognized in `Data2D`.
    """
    if isinstance(src, pd.DataFrame):
        return src
    if isinstance(src, np.ndarray):
        return pd.DataFrame(src, columns=columns)
    if isinstance(src, Tensor):
        return pd.DataFrame(src.numpy(), columns=columns)
    raise NotImplementedError(f'Unrecognized return type {type(src)}. '
                              f'Please make sure the input is one of [pd.DataFrame, np.ndarray, Tensor] type. ')
