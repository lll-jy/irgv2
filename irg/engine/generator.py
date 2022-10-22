"""Generate synthetic database."""

from collections import defaultdict
import os
from typing import Dict, Optional, DefaultDict, Any

import torch

from ..tabular import TabularTrainer
from ..schema import Table, SyntheticTable, Database, SyntheticDatabase
from ..schema.database import SYN_DB_TYPE_BY_NAME


def generate(real_db: Database, tab_models: Dict[str, TabularTrainer], deg_models: Dict[str, TabularTrainer],
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

    for name, table in real_db.tables():
        table = Table.load(table)
        table_temp_cache = os.path.join(temp_cache, name)
        if table.ttype == 'base':
            gen_table = table.shallow_copy()
            gen_table.update_temp_cache(table_temp_cache)
        elif table.is_independent():
            gen_table = _generate_independent_table(tab_models[name], table, scaling[name], tab_batch_sizes[name],
                                                    table_temp_cache)
        else:
            gen_table = _generate_dependent_table(tab_models[name], deg_models[name], table, scaling[name],
                                                  tab_batch_sizes[name], deg_batch_sizes[name], syn_db,
                                                  table_temp_cache)
        syn_db[name] = gen_table

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


def _generate_independent_table(trainer: TabularTrainer, table: Table, scale: float, batch_size: int, temp_cache: str) \
        -> Table:
    syn_table = SyntheticTable.from_real(table, temp_cache)
    need_rows = round(len(table) * scale)
    output = trainer.inference(torch.zeros(need_rows, 0), batch_size).output[:, -trainer.unknown_dim:].cpu()
    syn_table.inverse_transform(output, replace_content=True)
    return syn_table


def _generate_dependent_table(tab_trainer: TabularTrainer, deg_trainer: TabularTrainer, table: Table, scale: float,
                              tab_batch_size: int, deg_batch_size: int, syn_db: SyntheticDatabase, temp_cache: str) \
        -> SyntheticTable:
    print('start new generate', table.name)
    syn_table = SyntheticTable.from_real(table, temp_cache)
    print('generated real', flush=True)
    syn_db[table.name] = syn_table
    print('recorded dummy', flush=True)
    known = syn_db.degree_known_for(table.name)
    print('get degree known for', table.name, flush=True)
    deg_tensor = deg_trainer.inference(known, deg_batch_size).output[:, -deg_trainer.unknown_dim:].cpu()
    degrees = syn_table.inverse_transform_degrees(deg_tensor, scale)
    print('predicted degrees', degrees.describe())
    syn_table.assign_degrees(degrees)
    known_tab, _, _ = syn_table.ptg_data()
    output = tab_trainer.inference(known_tab, tab_batch_size).output[:, -tab_trainer.unknown_dim:].cpu()
    syn_table.inverse_transform(output, replace_content=True)
    print('inverse transformed dependent', table.name)
    print(syn_table.columns)
    syn_table.update_deg_and_aug()
    return syn_table
