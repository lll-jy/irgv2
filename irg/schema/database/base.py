"""Database schema definition."""

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from typing import OrderedDict as OrderedDictT, List, Optional, ItemsView, Any, Tuple, Dict
import os
import json

from jsonschema import validate
import pandas as pd
from torch import Tensor
from pandasql import sqldf

from ..table import Table, SyntheticTable
from ...utils.errors import ColumnNotFoundError, TableNotFoundError
from ...utils.misc import load_from, Data2D


class ForeignKey:
    """
    Foreign key helper structure.
    """
    def __init__(self, my_name: str, my_columns: List[str], parent_name: str, parent_columns: List[str]):
        """
        **Args**:

        - `my_name` (`str`): Child table's name.
        - `my_columns` (`List[str]`): Child table's columns in foreign key.
        - `parent_name` (`str`): Parent table's name.
        - `parent_columns` (`List[str]`): Parent table's columns in foreign key in order of `my_columns`).
        """
        self._name, self._parent = my_name, parent_name
        self._ref = {my_col: parent_col for my_col, parent_col in zip(my_columns, parent_columns)}

    @property
    def child(self) -> str:
        """Name of child table."""
        return self._name

    @property
    def parent(self) -> str:
        """Name of parent table."""
        return self._parent

    @property
    def ref(self) -> ItemsView[str, str]:
        """References as items view of child table columns to parent table columns in correspondence."""
        return self._ref.items()

    @property
    def left(self) -> List[Tuple[str, str]]:
        """When joining, left (child) column names as two-level name."""
        return [(self._name, col) for col in self._ref.keys()]

    @property
    def right(self) -> List[Tuple[str, str]]:
        """When joining, right (parent) column names as two-level name."""
        return [(self._parent, col) for col in self._ref.values()]


class Database(ABC):
    """Database data structure."""
    _TABLE_CONF = {
        'type': 'object',
        'properties': {
            'id_cols': {
                'type': 'array',
                'items': {'type': 'string'}
            },
            'attributes': {'type': 'object'},
            'path': {'type': 'string'},
            'format': {'enum': ['csv', 'pkl']},
            'determinants': {
                'type': 'array',
                'items': {'type': 'array', 'items': 'string'}
            },
            'formulas': {'type': 'object'},
            'primary_keys': {'type': 'array', 'items': 'string'},
            'foreign_keys': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'columns': {'type': 'array', 'items': 'string'},
                        'parent': {'type': 'string'},
                        'parent_columns': {'type': 'array', 'items': 'string'}
                    },
                    'required': ['columns', 'parent'],
                    'additionalProperties': False
                }
            }
        },
        'additionalProperties': False
    }

    def __init__(self, schema: OrderedDictT, data_dir: str = '.'):
        """
        **Args**:

        - `schema` (`OrderedDict`): Schema described as OrderedDict. Every table in the database corresponds to one
          entry in the dict, where the order of the dict is the order the tables are to be processed.
          Every table is described with key being its name, and value being another dict describing it, including the
          following content (all are optional but need to give enough information for constructing a
          [`Table`](../table#irg.schema.table.Table).
            - `id_cols`, `attributes`, `determinants`, `formulas`: arguments to
              [`Table`](../table#irg.schema.table.Table).
            - `path`: data file holding the content of this table under `data_dir`, and if not provided,
              it will be inferred from the table's name, and the path used will be data_dir/name.csv or pkl.
            - `format`: either 'csv' or 'pkl', depending on the file's format.
            - `primary_keys`: column names that constitute a primary key in the table.
            - `foreign_keys`: a list of foreign keys of the table, where each foreign key is described as
                - `columns`: list of column names in this table involved in the foreign key.
                - `parent`: name of parent table.
                - `parent_columns`: list of column names in the referenced (parent) table, in correspondence to
                  "columns" (in the same order), and will be assumed to be the same as "columns" if not provided.
        - `data_dir` (`str`): Directory of the data saved. It should typically contain all table paths
          as described in schema.

        **Raises**:

        - [`ColumnNotFoundError`](../../utils/errors#irg.utils.errors.ColumnNotFoundError) if some columns provided in
          some schema specification is not found in the corresponding table.
        - [`TableNotFoundError`](../../utils/errors#irg.utils.errors.TableNotFoundError) if some tables mentioned in
          the foreign keys as parents are not valid tables in the database (note that if the parent table appears after
          the child table, this error is also reported).
        - [`ValidationError`](https://python-jsonschema.readthedocs.io/en/stable/errors/) if the provided schema is of
          invalid format.
        """
        self._tables: OrderedDictT[str, Table] = OrderedDict({})
        self._primary_keys: Dict[str, List[str]] = {}
        self._foreign_keys: Dict[str, List[ForeignKey]] = {}
        self._data_dir = data_dir

        for name, meta in schema.items():
            validate(meta, self._TABLE_CONF)
            meta = defaultdict(lambda: None, meta)
            fm = meta['format'] if 'format' in meta else 'csv'
            path = meta['path'] if 'path' in meta else os.path.join(self._data_dir, f'{name}.{fm}')
            data = None
            if os.path.exists(path):
                data = pd.read_csv(path) if fm == 'csv' else pd.read_pickle(path)
            id_cols, attributes = meta['id_cols'], meta['attributes']
            determinants, formulas = meta['determinants'], meta['formulas']
            table = Table(
                name=name, need_fit=True,
                id_cols=id_cols, attributes=attributes, data=data,
                determinants=determinants, formulas=formulas
            )
            self._tables[name] = table
            columns = table.columns

            primary_keys = meta['primary_keys'] if 'primary_keys' in meta else columns
            columns = set(columns)
            for col in primary_keys:
                if col not in columns:
                    raise ColumnNotFoundError(name, col)
            self._primary_keys[name] = primary_keys
            foreign_keys = meta['foreign_keys'] if 'foreign_keys' in meta else []
            self._foreign_keys[name] = []
            for foreign_key in foreign_keys:
                this_columns = foreign_key['columns']
                parent_columns = foreign_key['parent_columns'] if 'parent_columns' in foreign_key else this_columns
                parent_name = foreign_key['parent']
                if parent_name not in self._tables:
                    raise TableNotFoundError(name)
                for col in this_columns:
                    if col not in columns:
                        raise ColumnNotFoundError(name, col)
                for col in parent_columns:
                    if col not in set(self._tables[parent_name].columns):
                        raise ColumnNotFoundError(parent_name, col)
                self._foreign_keys[name].append(ForeignKey(name, this_columns, parent_name, parent_columns))

            id_cols = [] if id_cols is None else id_cols
            determinants = [] if determinants is None else determinants
            formulas = {} if formulas is None else formulas
            for col in id_cols:
                if col not in columns:
                    raise ColumnNotFoundError(name, col)
            for determinant in determinants:
                for col in determinant:
                    if col not in columns:
                        raise ColumnNotFoundError(name, col)
            for col in formulas:
                if col not in columns:
                    raise ColumnNotFoundError(name, col)

    def __getitem__(self, item):
        return self._tables[item]

    def __len__(self):
        return len(self._tables)

    @classmethod
    def load_from(cls, file_path: str, engine: Optional[str] = None, data_dir: str = '.') -> "Database":
        """
        Load database from config file.

        **Args**:

        - `file_path` and `engine`: Arguments for [`utils.load_from`](../../utils/misc#irg.utils.misc.load_from)
        - `data_dir`: Argument for [constructor](#irg.schema.database.base.Database).
        """
        schema = load_from(file_path, engine)
        return Database(OrderedDict(schema), data_dir)

    @property
    @abstractmethod
    def mtype(self) -> str:
        """Mechanism type."""
        raise NotImplementedError()

    @abstractmethod
    def augment(self):
        """
        Augment the database according to the mechanism.
        """
        raise NotImplementedError()

    @property
    def tables(self) -> ItemsView[str, Table]:
        """Get all tables in list view."""
        return self._tables.items()

    def data(self, **kwargs) -> Dict[str, Data2D]:
        """
        Get data in desired format.

        **Args**:

        - `kwargs`: Arguments to [`Tables.data`](../table#irg.schema.table.Table.data).

        **Return**: A `dict` of names of tables mapped to `data` retrieval result of the table.
        """
        return {
            name: table.data(**kwargs)
            for name, table in self.tables
        }

    def query(self, query: str, descr: str = '', **kwargs) -> Table:
        """
        Execute SQL query in the database.

        **Args**:

        - `query` (`str`): Query to execute.
        - `descr` (`str`): Name of the queried table. Suggested to be a short description of the query.
        - `kwargs`: Other arguments to [`Table`](../table#irg.schema.table.Table) constructor.
          Argument `name` and `data` should not be passed.

        **Return**: Result of the queried table.
        """
        sqldb = self.data()
        query_data = sqldf(query, sqldb)
        query_table = Table(name=descr, data=query_data, **kwargs)
        return query_table

    def join(self, foreign_key: ForeignKey, descr: Optional[str] = None, how: str = 'outer') -> Table:
        """
        Join two tables using the foreign key.

        **Args**:
        - `foreign_key` (`ForeignKey`): The foreign key reference this joining is based on.
        - `descr` and `how`: Arguments to [`Table.join`](../table#irg.schema.table.Table.join).

        **Return**: The joined table.
        """
        child_table, parent_table = self[foreign_key.child], self[foreign_key.parent]
        return child_table.join(parent_table, foreign_key.ref, descr, how)

    def save_to_dir(self, path: str):
        """
        Save the database.

        - `path` (`str`): Path of the directory to save this database to. It is suggested to make sure that either the
          directory does not exist, the directory is empty, or all files inside are to be overwritten.
        """
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump({
                'primary_keys': self._primary_keys,
                'foreign_keys': self._foreign_keys,
                'data_dir': self._data_dir,
                'order': list(self._tables.keys())
            }, f, indent=2)

        for name, table in self.tables:
            table.save(os.path.join(path, f'{name}.pkl'))

    @property
    def all_joined(self) -> Table:
        data, id_cols, attributes, table_cnt = pd.DataFrame(), set(), {}, defaultdict(int)
        for name, table in self.tables:
            if data.empty or table.is_independent:
                if data.empty:
                    data = pd.concat({f'{name}_0': table.data()}, axis=1)
                else:
                    data = data.merge(pd.concat({f'{name}_0': table.data()}, axis=1), how='cross')
                id_cols |= {(f'{name}_0', col) for col in table.id_cols}
                attributes |= {(f'{name}_0', attr_name): attr for attr_name, attr in table.attributes.items()}
            else:
                for foreign_key in self._foreign_keys[name]:
                    for j in range(table_cnt[foreign_key.parent]+1):
                        left_on = [(f'{table}_{table_cnt[name]}', col) for table, col in foreign_key.left]
                        right_on = [(f'{table}_{j}', col) for table, col in foreign_key.right]
                        data = data.merge(pd.concat({f'{name}_{table_cnt[name]}': table.data()}, axis=1),
                                          how='outer', left_on=left_on, right_on=right_on)
                        id_cols |= {(f'{name}_{table_cnt[name]}', col) for col in table.id_cols}
                        attributes |= {(f'{name}_{table_cnt[name]}', attr_name): attr
                                       for attr_name, attr in table.attributes.items()}
                        table_cnt[name] += 1

        joined_table = Table(
            name='joined', need_fit=False, id_cols=id_cols, data=data
        )
        joined_table.replace_attributes(attributes)
        return joined_table

    @property
    def foreign_keys(self) -> List[ForeignKey]:
        """List of all foreign keys involved in this database."""
        return [key for name, keys in self._foreign_keys.items() for key in keys]

    @classmethod
    def load_from_dir(cls, path: str) -> "Database":
        """
        Load database from path.

        - `path` (`str`): Path of the directory of saved content of the database.

        **Return**: Loaded database.
        """
        database = Database(OrderedDict({}))

        with open(os.path.join(path, 'config.json'), 'r') as f:
            content = json.load(f)
        database._primary_keys, database._foreign_keys = content['primary_keys'], content['foreign_keys']
        database._data_dir, order = content['data_dir'], content['order']

        for table_name in order:
            database._tables[table_name] = Table.load(os.path.join(path, f'{table_name}.pkl'))

        return database

    @staticmethod
    @abstractmethod
    def _update_cls(item: Any):
        raise NotImplementedError()


class SyntheticDatabase(Database, ABC):
    """
    Synthetic database structure.
    """
    def __init__(self, real: Optional[Database] = None, **kwargs):
        """
        **Args**:

        - `real` (`Optional[Database]`): Real database that this synthetic database is trained on.
        - `kwargs`: Arguments for `Database`.
        """
        self._real = real
        super().__init__(**kwargs)

    @classmethod
    def from_real(cls, real_db: Database, save_to: str) -> "SyntheticDatabase":
        """
        Create synthetic database from real.

        **Args**:

        - `real_db` (`Database`): The real database.
        - `save_to` (`str`): Directory to save synthetically generated data.

        **Return**: Constructed empty synthetic database.
        """
        syn_db = SyntheticDatabase(real_db, schema=OrderedDict({}))
        syn_db._primary_keys, syn_db._foreign_keys = real_db._primary_keys, real_db._foreign_keys
        syn_db._data_dir = save_to
        os.makedirs(save_to, exist_ok=True)
        return syn_db

    @abstractmethod
    def degree_known_for(self, table_name: str) -> Tensor:
        """
        Get known tensor for degree generation.

        **Args**:

        - `table_name` (`str`): The name of degree table to get.

        **Return**: The degree model known part from already generated part.
        """
        raise NotImplementedError()

    def __setitem__(self, key: str, value: SyntheticTable):
        self._tables[key] = value

    def save_synthetic_data(self):
        """
        Save synthetic data to directory as CSV files.
        """
        for name, table in self.tables:
            table.data().to_csv(os.path.join(self._data_dir, f'{name}.csv'), index=False)

    def query(self, query: str, descr: str = '', **kwargs) -> SyntheticTable:
        real_result = self._real.query(query, descr, **kwargs)
        synthetic_result = super().query(query, descr, **kwargs)
        synthetic_table = SyntheticTable.from_real(real_result)
        synthetic_table.replace_data(synthetic_result.data())
        return synthetic_table
