"""IO util functions."""

import json
import os
import pickle
import sys
from typing import Any, Optional

import pandas as pd
import torch
from torch import load as torch_load
from torch.nn import Module
import yaml

from .misc import SparseDType
from .dist import is_main_process


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


def pd_to_pickle(df: pd.DataFrame, output_path: str, sparse: bool = True):
    """
    Save dataframe to pickle (compressed).

    **Args**:

    - `df` (`pd.DataFrame`): The dataframe to save.
    - `output_path` (`str`): File path to save.
    - `sparse` (`bool`): Whether to save the dataframe as sparse data. Default is `True`.
    """
    if sparse:
        df = df.astype(SparseDType).sparse
    pd.to_pickle(df, output_path,
                 compression={'method': 'gzip', 'compresslevel': 9})


def pd_read_compressed_pickle(file_path: str, sparse: bool = True) -> pd.DataFrame:
    """
    Read dataframe from pickle (compressed).

    **Args**:

    - `file_path` (`str`): Path of the file to load.
    - `sparse` (`bool`): Whether the saved dataframe is sparse data. Default is `True`.

    **Return**: Loaded dataframe.
    """
    loaded = pd.read_pickle(file_path, compression={'method': 'gzip', 'compresslevel': 9})
    if sparse:
        loaded = loaded.to_dense()
    return loaded


def save_state_dict(model: Module, path: str):
    """
    Save state dict of a model to the path.

    **Args**:

    - `model` (`Module`): The model to save.
    - `path` (`str`): Path to save the state dict to.
    """
    if is_main_process():
        if hasattr(model, 'module'):
            torch.save(model.module.state_dict(), path)
        else:
            torch.save(model.state_dict(), path)


def load_state_dict(model: Module, path: str) -> Module:
    """
    Load state dict of a model from the path.

    **Args**:

    - `model` (`Module`): The model to load.
    - `path` (`str`): Path the state dict was saved at.

    **Return**: Model with state dict loaded.
    """
    state_dict = torch.load(path)
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    return model


class HiddenPrints:
    """
    Helper class to hide print in some block.
    Adapted from https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print#:~:text=If%20you%20don't%20want,the%20top%20of%20the%20file.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
