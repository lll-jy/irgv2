"""Augment database."""

from typing import Optional, OrderedDict
import os
import logging

from ..schema import create_db, Database
from ..schema.database import DB_TYPE_BY_NAME

_LOGGER = logging.getLogger()


def augment(schema: Optional[OrderedDict] = None, file_path: Optional[str] = None, engine: Optional[str] = None,
            data_dir: str = '.', mtype: str = 'unrelated', save_db_to: Optional[str] = None, resume: bool = True,
            temp_cache: str = '.temp') \
        -> Database:
    """
    Augment database.

    **Args**:

    - `schema` to `mtype`: Arguments to [database creator](../schema/database#irg.schema.database.create)
    - `save_db_to` (`Optional[str]`): Save database to path.
    - `resume` (`bool`): Whether to use database saved at `save_db_to` or augmenting another time.
    - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.

    **Return**: Augmented database.
    """
    if save_db_to is not None and resume and os.path.exists(save_db_to):
        database = DB_TYPE_BY_NAME[mtype].load_from_dir(save_db_to)
        _LOGGER.info(f'Loaded database from {save_db_to}.')
        print(f'Loaded database from {save_db_to}.')
    else:
        os.makedirs(temp_cache, exist_ok=True)
        database = create_db(
            schema=schema,
            file_path=file_path,
            engine=engine,
            data_dir=data_dir,
            temp_cache=temp_cache,
            mtype=mtype
        )
        _LOGGER.info(f'Created database based on data in {data_dir}.')
        print(f'Created database based on data in {data_dir}.:: {mtype}')
        database.augment()
        _LOGGER.info('Augmented database.')
        if save_db_to is not None:
            database.save_to_dir(save_db_to)
            _LOGGER.info(f'Saved database to {save_db_to}.')
    return database
