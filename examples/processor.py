"""Base processor of databases."""

from abc import abstractmethod
import os
import json
from types import FunctionType
from typing import Dict, List, Optional

import pandas as pd


class DatabaseProcessor:
    """Processor of databases."""
    def __init__(self, name: str, src_data_dir: str, data_dir: str, meta_dir: str, tables: List[str], out: str):
        """
        **Args**:

        - `name` (`str`): Name of the database.
        - `src_data_dir` (`str`): Directory holding source (unprocessed) data files.
        - `data_dir` (`str`): Directory to output processed table data.
        - `meta_dir` (`str`): Directory holding the metadata of the tables.
        - `tables` (`List[str]`): List of tables to process.
        - `out` (`str`): File path of the database config file.
        """
        self._name, self._src_data_dir, self._data_dir, self._meta_dir = name, src_data_dir, meta_dir, data_dir
        self._tables, self._out = tables, out

        os.makedirs(self._data_dir, exist_ok=True)
        os.makedirs(self._meta_dir, exist_ok=True)

    @property
    def name(self) -> str:
        """Name of the database."""
        return self._name

    def process_table_data(self, table_name: str, src: Optional[str] = None, out: Optional[str] = None):
        """
        Process data of a table.

        **Args**:

        - `table_name` (`str`): Name of the table to be processed.
        - `src` (`Optional[str]`): Source data path.
        - `out` (`Optional[str]`): Output data path. If not provided, then save at `data_dir/table_name.pkl`.
          If the provided path is a directory, the file is saved under the directory by the name of the table name as a
          `.pkl` file.
        """
        src = pd.read_csv(src if src is not None else
                          os.path.join(self._src_data_dir, f'{self._table_src_name_map[table_name]}.csv'),
                          encoding=self._source_encoding)
        out = out if out is not None else self._data_dir
        out = os.path.join(out, f'{table_name}.pkl') if os.path.isdir(out) else out
        processed: pd.DataFrame = self._table_data_processors[table_name](src)
        processed.to_pickle(out)

    def construct_table_metadata(self, table_name: str, src: Optional[str] = None, out: Optional[str] = None):
        """
        Construct metadata for a table.

        **Args**:

        - `table_name` (`str`): Name of the table whose metadata is to be constructed.
        - `src` (`Optional[str]`): Path of the processed table data. If not provided, the table is found from data_dir.
        - `out` (`Optional[str]`): Path to output the metadata JSON file. If not provided, the result is saved at
          `meta_dir/table_name.json`.
        """
        src = src if src is not None else os.path.join(self._data_dir, f'{table_name}.pkl')
        out = out if out is not None else self._meta_dir
        out = os.path.join(out, f'{table_name}.json') if os.path.isdir(out) else out
        src = pd.read_pickle(src)
        constructed = self._table_metadata_constructors[table_name](src)
        with open(out, 'w') as f:
            json.dump(constructed, f, indent=2)

    @abstractmethod
    @property
    def _table_metadata_constructors(self) -> Dict[str, FunctionType]:
        raise NotImplementedError()

    @abstractmethod
    @property
    def _table_src_name_map(self) -> Dict[str, str]:
        raise NotImplementedError()

    @abstractmethod
    @property
    def _table_data_processors(self) -> Dict[str, FunctionType]:
        raise NotImplementedError()

    @abstractmethod
    @property
    def _source_encoding(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def postprocess(self, output_dir: Optional[str] = None, sample: Optional[int] = None):
        """
        Post-process taking the entire database's data into consideration.
        For example, remove records of students whose information is incomplete.
        Such post-process usually involve multiple tables.

        **Args**:

        - `output_dir` (`Optional[str]`): Output path of the post-processing step.
          If not provided, it will override the data_dir data.
        - `sample` (`Optional[int]`): If not None, sample this number of basic form (e.g. students for ALSET) of data.
        """
        raise NotImplementedError()

    @classmethod
    def create_dummy(cls) -> "DatabaseProcessor":
        """
        Create a dummy processor, typically for the purpose of processing only one or two tables of the database.

        **Return**:

        Created processor. Only applies to child classes of `DatabaseProcessor`.
        """
        return cls('', '', '', '')

    def process_database(self, redo_meta: bool = True, redo_data: bool = True):
        """
        Process all tables in the database.

        **Args**:

        - `redo_meta` (`bool`): Whether to force reconstructing metadata.
        - `redo_data` (`bool`): Whether to force reprocessing data.
        """
        out_config = {}
        for table_name in self._tables:
            meta_path = os.path.join(self._meta_dir, f'{table_name}.json')
            if not os.path.exists(meta_path) or redo_meta:
                if self._data_dir is None:
                    raise ValueError('Need to access processed data, please provide data_dir.')
                data_path = os.path.join(self._data_dir, f'{table_name}.pkl')
                if not os.path.exists(data_path) or redo_data:
                    self.process_table_data(table_name)
                self.construct_table_metadata(table_name)
            with open(meta_path, 'r') as f:
                loaded = json.load(f)
                loaded['format'] = 'pickle'
                out_config[table_name] = loaded

        self.postprocess()

        with open(self._out, 'w') as f:
            json.dump(out_config, f, indent=2)
