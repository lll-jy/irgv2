"""Database structure and augmenting helpers."""

from typing import OrderedDict, Dict, Optional

from .base import Database, SyntheticDatabase
from .unrelated import UnrelatedDatabase
from .parent import ParentChildDatabase
from .ancestor import AncestorDescendantDatabase
from .affecting import AffectingDatabase


__all__ = (
    'Database',
    'SyntheticDatabase',
    'UnrelatedDatabase',
    'ParentChildDatabase',
    'AncestorDescendantDatabase',
    'AffectingDatabase',
    'create',
    'create_from_dict',
    'create_from_file'
)

_DB_TYPE_BY_NAME: Dict[str, Database.__class__] = {
    'unrelated': UnrelatedDatabase,
    'parent-child': ParentChildDatabase,
    'ancestor-descendant': AncestorDescendantDatabase,
    'affecting': AffectingDatabase
}


def create(schema: Optional[OrderedDict] = None, file_path: Optional[str] = None, engine: Optional[str] = None,
           data_dir: str = '.', temp_cache: str = '.temp', mtype: str = 'affecting') -> Database:
    """
    Create database from schema.

    **Args**: Arguments to `create_from_dict` and `create_from_file`. Apply whichever sufficient information is given.
    If both are applicable, `create_from_file` is favored.

    **Return**: Constructed database from the given information.

    **Raises**: `RuntimeError` if neither `create_from_dict` nor `create_from_file` is applicable.
    """
    if file_path is not None:
        return create_from_file(file_path, engine, data_dir, mtype)
    if schema is not None:
        return create_from_dict(schema, data_dir, temp_cache, mtype)
    raise RuntimeError('Schema and file path cannot be None simultaneously.')


def create_from_dict(schema: OrderedDict, data_dir: str = '.', temp_cache: str = '.temp', mtype: str = 'affecting') \
        -> Database:
    """
    Create database from schema as dict.

    **Args**:

    - `schema`, `data_dir`, and `temp_cache`: Arguments to [`Database`](./base#irg.schema.database.base.Database)
      constructor.
    - `mtype` (`str`): Mechanism type. Supported types are 'unrelated', 'parent-child', 'ancestor-descendant',
      and 'affecting'. Default is 'affecting'.

    **Return**: Constructed database from the given information.
    """
    return _DB_TYPE_BY_NAME[mtype](schema, data_dir, temp_cache)


def create_from_file(file_path: str, engine: Optional[str] = None, data_dir: str = '.', mtype: str = 'affecting') \
        -> Database:
    """
    Create database from schema in file.

    **Args**:

    - `file_path`, `engine` and `data_dir`: Arguments to
      [`Database.load_from`](./base#irg.schema.database.base.Database.load_from).
    - `mtype` (`str`): Mechanism type. Supported types are 'unrelated', 'parent-child', 'ancestor-descendant',
      and 'affecting'. Default is 'affecting'.

    **Return**: Constructed database from the given information.
    """
    return _DB_TYPE_BY_NAME[mtype].load_from(file_path, engine, data_dir)
