"""Miscellaneous util functions."""

from typing import Optional, Union, Collection, Literal
from statistics import harmonic_mean
from datetime import datetime

import numpy as np
import pandas as pd
from torch import Tensor, from_numpy as tensor_from_numpy
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from .dist import is_initialized, get_device

__all__ = (
    'Data2D',
    'Data2DName',
    'SparseDType',
    'convert_data_as',
    'inverse_convert_data',
    'calculate_mean',
    'make_dataloader',
    'reformat_datetime'
)


Data2D = Union[pd.DataFrame, np.ndarray, Tensor]
"""2D data type, including `pd.DataFrame`, `np.ndarray`, and `torch.Tensor`."""
SparseDType = pd.SparseDtype('float32', fill_value=0)
"""Sparse data type for general usage."""
Data2DName = Literal['pandas', 'numpy', 'torch']
"""
2D data type name. Literal of `pandas`, `numpy`, and `tensor`.

- `pandas`: `pd.DataFrame`.
- `numpy`: `np.ndarray`.
- `tensor`: `torch.Tensor`.
"""


def convert_data_as(src: pd.DataFrame, return_as: Data2DName = 'pandas', copy: bool = True) -> Data2D:
    """
    Convert a pd.DataFrame to desired data type.

    **Args**:

    - `src` (`pd.DataFrame`): The data to be converted.
    - `return_as` (`Data2DName`) [default 'pandas']: Data type to return.
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


def calculate_mean(x: Union[pd.Series, np.ndarray, Tensor], mean: str = 'arithmetic', smooth: float = 0.1) -> float:
    """
    **Args**:

    - `x` (`Union[pd.Series, np.ndarray, Tensor]`): Data to calculate mean value.
    - `mean` (`str`): The way mean is calculated. Can be either 'arithmetic' or 'harmonic'. Default is 'arithmetic'.
    - `smooth` (`float`): Smoothing value in case of zero division when calculating harmonic mean.
      The harmonic value calculated will be HM(x + smooth) - smooth, and when smooth = 0, it is not smoothed.

    **Return**: The calculated mean value.
    """
    if mean not in {'arithmetic', 'harmonic'}:
        raise NotImplementedError(f'Mean by {mean} is not implemented. Please give "arithmetic" or "harmonic".')
    if not 0 <= smooth <= 1:
        raise ValueError(f'Smooth value should be in [0, 1], got {smooth}.')

    x = np.array([y for y in x if not pd.isnull(y)])
    if len(x) == 0:
        return np.nan

    if mean == 'arithmetic':
        return x.mean()
    else:
        return harmonic_mean(x + smooth) - smooth


def reformat_datetime(x: Optional[datetime], format_str: str) -> Optional[datetime]:
    """
    Reformat datetime to desired format.

    **Args**:

    - `x` (`Optional[datetime]`): The original datetime.
    - `format_str` (`str`): The target new format.

    **Return**: Reformatted datetime.
    """
    if pd.isnull(x):
        return x
    return datetime.strptime(x.strftime(format_str), format_str)


def make_dataloader(*x: Tensor, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    """
    Make dataloader for training based the data.

    **Args**:

    - `x` (`Tensor`): Data to build loader from.
    - `batch_size` (`int`): Batch size of the data loader. Default is 64.
    - `shuffle` (`bool`): Whether to shuffle.
    """
    dataset = [t.to(get_device()) for t in x]
    dataset = TensorDataset(*dataset)
    if is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    elif shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=5,
        pin_memory=True
    )
    return loader
