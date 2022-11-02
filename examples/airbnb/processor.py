"""Data processor for Airbnb database."""
from typing import Dict, Callable, Optional, List
import os
import shutil

import pandas as pd

from ..processor import DatabaseProcessor
from . import data, metadata

AIRBNB_PROCESSORS: Dict[str, Callable] = {
    'countries': data.countries,
    'age_gender': data.age_gender,
    'users': data.users,
    'sessions': data.sessions
}
"""Airbnb table data processors."""

AIRBNB_META_CONSTRUCTORS: Dict[str, Callable] = {
    'countries': metadata.countries,
    'age_gender': metadata.age_gender,
    'users': metadata.users,
    'sessions': metadata.sessions
}
"""Airbnb metadata constructors for each table."""

AIRBNB_PROCESS_NAME_MAP: Dict[str, str] = {
    'countries': 'downloaded/countries',
    'age_gender': 'downloaded/age_gender_bkts',
    'users': 'downloaded/train_users_2',
    'sessions': 'downloaded/sessions'
}
"""Airbnb source data file names (without extension) for all tables."""


class AirbnbProcessor(DatabaseProcessor):
    """Data processor for Airbnb database."""
    def __init__(self, src_data_dir: str, data_dir: str, meta_dir: str, out: str, tables: Optional[List[str]] = None):
        """
        **Args**:

        Arguments to `DatabaseProcessor`.
        If tables is not specified, all recognized tables are processed.
        """
        if tables is None or len(tables) == 0:
            tables = [*AIRBNB_PROCESSORS]
        super().__init__('airbnb', src_data_dir, data_dir, meta_dir, tables, out)

    @property
    def _table_data_processors(self) -> Dict[str, Callable]:
        return AIRBNB_PROCESSORS

    @property
    def _table_src_name_map(self) -> Dict[str, str]:
        return AIRBNB_PROCESS_NAME_MAP

    @property
    def _table_metadata_constructors(self) -> Dict[str, Callable]:
        return AIRBNB_META_CONSTRUCTORS

    @property
    def _source_encoding(self) -> Optional[str]:
        return None

    def postprocess(self, output_dir: Optional[str] = None, sample: Optional[int] = None):
        if sample is not None:
            if 'users' not in self._tables:
                raise ValueError('Cannot sample ALSET database without personal data table.')
            if output_dir is None:
                raise ValueError('Cannot sample without specifying output directory.')
            os.makedirs(output_dir, exist_ok=True)
            users = pd.read_table(os.path.join(self._data_dir, 'users.pkl'))
            selected_users = users['id'].sample(sample)
            if 'sessions' in self._tables:
                sessions = pd.read_pickle(os.path.join(self._data_dir, 'sessions.pkl'))
                sessions = sessions[sessions['user_id'].isin(selected_users)].reset_index(drop=True)
                sessions.to_pickle(os.path.join(output_dir, 'sessions.pkl'))
        elif output_dir is not None:
            shutil.copytree(self._data_dir, output_dir, dirs_exist_ok=True)

    
