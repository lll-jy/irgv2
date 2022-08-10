"""Generate synthetic database."""

from collections import defaultdict
from typing import Dict, Optional, DefaultDict, Any

import torch

from ..schema import Table, SyntheticTable, Database, SyntheticDatabase
from ..utils import Trainer


def generate(real_db: Database, tab_models: Dict[str, Trainer], deg_models: Dict[str, Trainer], save_to: str,
             scaling: Optional[Dict[str, float]] = None,
             tab_batch_sizes: Optional[Dict[str, int]] = None, deg_batch_sizes: Optional[Dict[str, int]] = None,
             save_db_to: Optional[str] = None) -> SyntheticDatabase:
    """
    Generate synthetic database.

    **Args**:
    - `real_db` (`Database`): Real database.
    - `tab_models` (`Dict[str, Trainer]`): Trainers for tabular data for each table.
    - `deg_models` (`Dict[str, Trainer]`): Trainers for degree data for each table.
    - `save_to` (`str`): Save the generated tables to path.
    - `scaling` (`Optional[Dict[str, float]]`): Scaling factors of synthetic data. Default scaling factor is 1.
    - `tab_batch_sizes` (`Optional[Dict[str, int]]`): Batch size when running inference for tabular models.
      Default is 32.
    - `deg_batch_sizes` (`Optional[Dict[str, int]]`): Batch size when running inference for degree models.
      Default is 32.
    - `save_db_to` (`Optional[str]`): Save the synthetic database to directory. If `None`, do not save.

    **Return**: Synthetically generated database.
    """
    syn_db = SyntheticDatabase.from_real(real_db, save_to)
    scaling = _optional_default_dict(scaling, 1.)
    tab_batch_sizes = _optional_default_dict(tab_batch_sizes, 32)
    deg_batch_sizes = _optional_default_dict(deg_batch_sizes, 32)

    for name, table in real_db.tables:
        if table.is_independent:
            gen_table = _generate_independent_table(tab_models[name], table, scaling[name], tab_batch_sizes[name])
        else:
            gen_table = _generate_dependent_table(tab_models[name], deg_models[name], table, scaling[name],
                                                  tab_batch_sizes[name], deg_batch_sizes[name], syn_db)
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


def _generate_independent_table(trainer: Trainer, table: Table, scale: float, batch_size: int) -> Table:
    syn_table = SyntheticTable.from_real(table)
    need_rows = round(len(table) * scale)
    output = trainer.inference(torch.zeros(need_rows, 0), batch_size).output
    syn_table.inverse_transform(output)
    return syn_table


def _generate_dependent_table(tab_trainer: Trainer, deg_trainer: Trainer, table: Table, scale: float,
                              tab_batch_size: int, deg_batch_size: int, syn_db: SyntheticDatabase) -> SyntheticTable:
    syn_table = SyntheticTable.from_real(table)
    known = syn_db.degree_known_for(table.name)
    deg_tensor = deg_trainer.inference(known, deg_batch_size).output
    degrees = syn_table.inverse_transform_degrees(deg_tensor, scale)
    syn_table.assign_degrees(degrees)
    known_tab, _, _ = syn_table.ptg_data
    output = tab_trainer.inference(known_tab, tab_batch_size).output
    syn_table.inverse_transform(output)
    return syn_table
