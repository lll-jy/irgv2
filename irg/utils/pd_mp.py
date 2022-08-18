"""Pandas functions by multiprocessing."""
import math
from types import FunctionType
from typing import Union, Any

import numpy as np
import pandas as pd

from .dist import _pool, fast_map

PandasData = Union[pd.Series, pd.DataFrame]
"""Pandas series or dataframe."""


def _wrapper(func: FunctionType, data: PandasData, chunk_size: int = 100, **kwargs) -> PandasData:
    if _pool is None:
        return func(data, **kwargs)
    split = np.array_split(data, math.ceil(len(data) / chunk_size))
    fragments = fast_map(
        func=func,
        iterable=split,
        total_len=len(split),
        func_kwargs=kwargs
    )
    result = pd.concat(fragments).reset_index(drop=True)
    result.index = data.index
    return result


def fillna(data: PandasData, chunk_size: int = 100, **kwargs) -> PandasData:
    return _wrapper(data.__class__.fillna, data, chunk_size, **kwargs)
