import json
import pickle
from typing import Optional, Union, Collection, Any
import yaml
from statistics import harmonic_mean
from datetime import datetime

import numpy as np
import pandas as pd
from torch import Tensor, from_numpy as tensor_from_numpy, load as torch_load


__all__ = (
    'Data2D',
    'convert_data_as',
    'inverse_convert_data',
    'load_from',
    'calculate_mean'
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


def load_from(file_path: str, engine: Optional[str] = None) -> Any:
    """
    Load content from file.

    **Args**:

    - `file_path` (`str`): Path to the file.
    - `engine` (`Optional[str]`) [default `None`]: File format. Supported format include json, pickle, yaml, and torch.
      If `None`, infer from the extension name of the file path.

    **Return**: Content of the file.

    **Raise**: `NotImplementedError` if the engine is not recognized.
    """
    if engine is None:
        if file_path.endswith('.json'):
            engine = 'json'
        elif file_path.endswith('.pkl'):
            engine = 'pickle'
        elif file_path.endswith('.yaml'):
            engine = 'yaml'
        elif file_path.endswith('.pt') or file_path.endswith('.bin') or file_path.endswith('.pth'):
            engine = 'torch'

    if engine == 'json':
        with open(file_path, 'r') as f:
            content = json.load(f)
        return content
    if engine == 'pickle':
        with open(file_path, 'rb') as f:
            content = pickle.load(f)
        return content
    if engine == 'yaml':
        with open(file_path, 'r') as f:
            content = yaml.load(f, Loader=yaml.FullLoader)
        return content
    if engine == 'torch':
        return torch_load(file_path)
    raise NotImplementedError(f'Data file of {engine} is not recognized as a valid engine.')


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
