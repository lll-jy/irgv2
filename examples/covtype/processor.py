import os
import shutil
from typing import Any, Callable, Dict, Optional, List
from types import FunctionType

import numpy as np
import pandas as pd

from irg.schema import Table

from ..processor import DatabaseProcessor


COVTYPE_PROCESSORS: Dict[str, Callable[pd.DataFrame, pd.DataFrame]] = {
    'covtype': lambda x: x
}
"""Covtype table data processors."""


def _covtype_metadata(src: pd.DataFrame) -> Dict[str, Any]:
    attributes = Table.learn_meta(src)
    return {'attributes': attributes}


COVTYPE_META_CONSTRUCTORS: Dict[str, Callable[pd.DataFrame, Dict[str, Any]]] = {
    'covtype': _covtype_metadata
}
"""Covtype table metadata constructor"""

COVTYPE_PROCESS_NAME_MAP: Dict[str, str] = {
    'covtype': 'covtype.train'
}
"""Covtype table source data file"""


class CovtypeProcessor(DatabaseProcessor):
    """Data processor for Covtype dataset."""
    def __init__(self, src_data_dir: str, data_dir: str, meta_dir: str, out: str, tables: Optional[List[str]] = None):
        """
        **Args**:

        Arguments to `DatabaseProcessor`.
        """
        tables = ['covtype']
        super().__init__('covtype', src_data_dir, data_dir, meta_dir, ['covtype'], out)

    @property
    def _table_data_processors(self) -> Dict[str, FunctionType]:
        return COVTYPE_PROCESSORS

    @property
    def _table_src_name_map(self) -> Dict[str, str]:
        return COVTYPE_PROCESS_NAME_MAP

    @property
    def _table_metadata_constructors(self) -> Dict[str, FunctionType]:
        return COVTYPE_META_CONSTRUCTORS

    @property
    def _source_encoding(self) -> Optional[str]:
        return None

    def postprocess(self, output_dir: Optional[str] = None, sample: Optional[int] = None):
        if sample is not None:
            data = pd.read_pickle(os.path.join(self._data_dir, 'covtype.pkl'))
            sampled_ids = np.random.choice(range(len(data)), sample, replace=False)
            sampled = data.iloc[sampled_ids].reset_index(drop=True)
            sampled.to_pickle(os.path.join(output_dir, 'covtype.pkl'))
        elif output_dir is not None:
            shutil.copytree(self._data_dir, output_dir, dirs_exist_ok=True)
