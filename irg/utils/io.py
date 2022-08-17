"""IO util functions."""

import json
import pickle
from typing import Optional, Any

import yaml
from torch import load as torch_load
import pandas as pd


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


def pd_to_pickle(df: pd.DataFrame, output_path: str):
    """
    Save dataframe to pickle.

    **Args**:

    - `df` (`pd.DataFrame`): The dataframe to save.
    - `output_path` (`str`): File path to save.
    """
    df.to_pickle(output_path,
                 compression={'method': 'gzip', 'compresslevel': 9})


def pd_read_compressed_pickle(file_path: str) -> pd.DataFrame:
    """
    Read dataframe from pickle (compressed).

    **Args**:

    - `file_path` (`str`): Path of the file to load.

    **Return**: Loaded dataframe.
    """
    return pd.read_pickle(file_path, compression={'method': 'gzip', 'compresslevel': 9})
