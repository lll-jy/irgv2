"""Pandas functions by multiprocessing."""
import math
from types import FunctionType
from typing import Union

import numpy as np
import pandas as pd

from .dist import pool_initialized, fast_map

PandasData = Union[pd.Series, pd.DataFrame]
"""Pandas series or dataframe."""


def _wrapper(func: FunctionType, data: PandasData, chunk_size: int = 100, **kwargs) -> PandasData:
    if not pool_initialized():
        return data.__class__(func(data, **kwargs))
    split = np.array_split(data, math.ceil(len(data) / chunk_size))
    fragments = fast_map(
        func=func,
        iterable=split,
        total_len=len(split),
        func_kwargs=kwargs
    )
    if len(fragments) == 0:
        if isinstance(data, pd.Series):
            return pd.Series()
        else:
            return pd.DataFrame()

    if isinstance(fragments[0], np.ndarray):
        result = data.__class__(np.hstack(fragments))
    else:
        result = pd.concat(fragments).reset_index(drop=True)
    result.index = data.index
    return result


def fillna(data: PandasData, chunk_size: int = 100, **kwargs) -> PandasData:
    return _wrapper(data.__class__.fillna, data, chunk_size, **kwargs)


def unique(data: PandasData, chunk_size: int = 100, **kwargs) -> PandasData:
    return _wrapper(data.__class__.unique, data, chunk_size, **kwargs)
