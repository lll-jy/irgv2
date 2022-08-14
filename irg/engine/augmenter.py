"""Augment database."""

from typing import Optional, OrderedDict
import os
import logging

from ..schema import create_db, Database

_LOGGER = logging.getLogger()


def augment(schema: Optional[OrderedDict] = None, file_path: Optional[str] = None, engine: Optional[str] = None,
            data_dir: str = '.', mtype: str = 'unrelated', save_db_to: Optional[str] = None, resume: bool = True) \
        -> Database:
    """
    Augment database.

    **Args**:

    - `schema` to `mtype`: Arguments to [database creator](../schema/database#irg.schema.database.create)
    - `save_db_to` (`Optional[str]`): Save database to path.
    - `resume` (`bool`): Whether to use database saved at `save_db_to` or augmenting another time.

    **Return**: Augmented database.
    """
    if save_db_to is not None and resume and os.path.exists(save_db_to):
        database = Database.load_from_dir(save_db_to)
        _LOGGER.info(f'Loaded database from {save_db_to}.')
    else:
        database = create_db(schema, file_path, engine, data_dir, mtype)
        _LOGGER.info(f'Created database based on data in {data_dir}.')
        database.augment()
        _LOGGER.info('Augmented database.')
        if save_db_to is not None:
            database.save_to_dir(save_db_to)
            _LOGGER.info(f'Saved database to {save_db_to}.')
    return database
