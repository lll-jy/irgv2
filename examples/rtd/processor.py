"""Data processor for RTD database."""

from typing import List, Dict, Optional
from types import FunctionType
import shutil

from ..processor import DatabaseProcessor
from . import data, metadata


RTD_PROCESSORS: Dict[str, FunctionType] = {
    'country': data.country,
    'provstate': data.provstate,
    'city': data.city,
    'events': data.events,
    'life_damage': data.life_damage,
    'eco_damage': data.eco_damage,
    'hostkid': data.hostkid,
    'info_int': data.info_int,
    'attack_type': data.attack_type,
    'target': data.target,
    'group': data.group,
    'claim': data.claim,
    'weapon': data.weapon,
    'related': data.related
}
"""RTD table data processors."""

RTD_META_CONSTRUCTORS: Dict[str, FunctionType] = {
    'country': metadata.country,
    'provstate': metadata.provstate,
    'city': metadata.city,
    'events': metadata.events,
    'life_damage': metadata.life_damage,
    'eco_damage': metadata.eco_damage,
    'hostkid': metadata.hostkid,
    'info_int': metadata.info_int,
    'attack_type': metadata.attack_type,
    'target': metadata.target,
    'group': metadata.group,
    'claim': metadata.claim,
    'weapon': metadata.weapon,
    'related': metadata.related
}
"""RTD metadata constructors for each table."""

RTD_PROCESS_NAME_MAP: Dict[str, str] = {
    'country': 'rtd',
    'provstate': 'rtd',
    'city': 'rtd',
    'events': 'rtd',
    'life_damage': 'rtd',
    'eco_damage': 'rtd',
    'hostkid': 'rtd',
    'info_int': 'rtd',
    'attack_type': 'rtd',
    'target': 'rtd',
    'group': 'rtd',
    'claim': 'rtd',
    'weapon': 'rtd',
    'related': 'rtd'
}
"""RTD source data file names (without extension) for all tables."""


class RTDProcessor(DatabaseProcessor):
    """Data processor for RTD database."""
    def __init__(self, src_data_dir: str, data_dir: str, meta_dir: str, out: str, tables: Optional[List[str]]):
        """
        **Args**:

        Arguments to `DatabaseProcessor`.
        If tables is not specified, all recognized tables are processed.
        """
        if tables is None:
            tables = [*RTD_PROCESSORS]
        super().__init__('rtd', src_data_dir, data_dir, meta_dir, tables, out)

    @property
    def _source_encoding(self) -> str:
        return 'ISO-8859-1'

    @property
    def _table_data_processors(self) -> Dict[str, FunctionType]:
        return RTD_PROCESSORS

    @property
    def _table_src_name_map(self) -> Dict[str, str]:
        return RTD_PROCESS_NAME_MAP

    @property
    def _table_metadata_constructors(self) -> Dict[str, FunctionType]:
        return RTD_META_CONSTRUCTORS

    def postprocess(self, output_dir: Optional[str] = None):
        if output_dir is not None:
            shutil.copytree(self._data_dir, output_dir, dirs_exist_ok=True)
