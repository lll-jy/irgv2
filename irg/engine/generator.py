"""Generate synthetic database."""

from collections import defaultdict
import os
from typing import Dict, Optional, DefaultDict, Any

import torch

from ..degree import DegreeTrainer
from ..tabular import TabularTrainer
from ..schema import Table, SyntheticTable, Database, SyntheticDatabase, SeriesTable
from ..schema.database import SYN_DB_TYPE_BY_NAME


def generate(real_db: Database, tab_models: Dict[str, TabularTrainer], deg_models: Dict[str, DegreeTrainer],
             save_to: str, scaling: Optional[Dict[str, float]] = None,
             tab_batch_sizes: Optional[Dict[str, int]] = None, deg_batch_sizes: Optional[Dict[str, int]] = None,
             save_db_to: Optional[str] = None, temp_cache: str = '.temp') -> SyntheticDatabase:
    """
    Generate synthetic database.

    **Args**:
    - `real_db` (`Database`): Real database.
    - `tab_models` (`Dict[str, TabularTrainer]`): Trainers for tabular data for each table.
    - `deg_models` (`Dict[str, TabularTrainer]`): Trainers for degree data for each table.
    - `save_to` (`str`): Save the generated tables to path.
    - `scaling` (`Optional[Dict[str, float]]`): Scaling factors of synthetic data. Default scaling factor is 1.
    - `tab_batch_sizes` (`Optional[Dict[str, int]]`): Batch size when running inference for tabular models.
      Default is 32.
    - `deg_batch_sizes` (`Optional[Dict[str, int]]`): Batch size when running inference for degree models.
      Default is 32.
    - `save_db_to` (`Optional[str]`): Save the synthetic database to directory. If `None`, do not save.
    - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.

    **Return**: Synthetically generated database.
    """
    os.makedirs(temp_cache, exist_ok=True)
    temp_cache = os.path.join(temp_cache, 'generated')
    syn_db = SYN_DB_TYPE_BY_NAME[real_db.mtype].from_real(real_db, save_to)
    scaling = _optional_default_dict(scaling, 1.)
    tab_batch_sizes = _optional_default_dict(tab_batch_sizes, 32)
    deg_batch_sizes = _optional_default_dict(deg_batch_sizes, 32)
    os.makedirs(save_db_to, exist_ok=True)

    for name, table in real_db.tables():
        if real_db.is_series(name):
            table_class = Table
        else:
            table_class = SeriesTable
        table = table_class.load(table)
        table_temp_cache = os.path.join(temp_cache, name)
        if os.path.exists(os.path.join(save_db_to, f'{name}.pkl')):
            gen_table = table_class.load(os.path.join(save_db_to, f'{name}.pkl'))
        elif table.ttype == 'base':
            gen_table = _generate_base_table(table, scaling[name], table_temp_cache)
        elif table.is_independent():
            gen_table = _generate_independent_table(tab_models[name], table, scaling[name], tab_batch_sizes[name],
                                                    table_temp_cache)
        else:
            gen_table = _generate_dependent_table(tab_models[name], deg_models[name], table, scaling,
                                                  tab_batch_sizes[name], deg_batch_sizes[name], syn_db,
                                                  table_temp_cache)
        print('finished generation', name)
        syn_db[name] = gen_table
        print('updated database')
        gen_table.save(os.path.join(save_db_to, f'{name}.pkl'))

    syn_db.save_synthetic_data()
    if save_db_to is not None:
        syn_db.save_to_dir(save_db_to)
    return syn_db


def _optional_default_dict(original: Optional[Dict], default_val: Any) -> DefaultDict:
    if original is None:
        return defaultdict(lambda: default_val)
    if not isinstance(original, DefaultDict):
        return defaultdict(lambda: default_val, default_val)
    return original


def _generate_base_table(table: Table, scale: float, temp_cache: str) -> Table:
    syn_table = SyntheticTable.from_real(table, temp_cache)
    need_rows = round(len(table) * scale)
    syn_table.replace_data(table.data().sample(n=need_rows, replace=need_rows > len(table)).reset_index(drop=True))
    return syn_table


def _generate_independent_table(trainer: TabularTrainer, table: Table, scale: float, batch_size: int, temp_cache: str) \
        -> Table:
    syn_table = SyntheticTable.from_real(table, temp_cache)
    need_rows = round(len(table) * scale)
    output = trainer.inference(torch.zeros(need_rows, 0), batch_size).output[:, -trainer.unknown_dim:].cpu()
    syn_table.inverse_transform(output, replace_content=True)
    return syn_table


def _generate_dependent_table(tab_trainer: TabularTrainer, deg_trainer: DegreeTrainer, table: Table,
                              scaling: Dict[str, float], tab_batch_size: int, deg_batch_size: int,
                              syn_db: SyntheticDatabase, temp_cache: str) \
        -> SyntheticTable:
    syn_table = SyntheticTable.from_real(table, temp_cache)
    syn_db.save_dummy(table.name, syn_table)

    known_tab, augmented = deg_trainer.predict(syn_table, syn_db, scaling)
    syn_table.update_augmented(augmented)

    output = tab_trainer.inference(known_tab, tab_batch_size)
    syn_table.inverse_transform(output, replace_content=True)
    syn_table.update_deg_and_aug()
    return syn_table
