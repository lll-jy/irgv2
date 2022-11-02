"""Generate synthetic database."""

from collections import defaultdict
import os
from typing import Dict, Optional, DefaultDict, Any

import pandas as pd
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
    print('--get started --', flush=True)
    syn_db = SYN_DB_TYPE_BY_NAME[real_db.mtype].from_real(real_db, save_to)
    print('crated syn', flush=True)
    scaling = _optional_default_dict(scaling, 1.)
    tab_batch_sizes = _optional_default_dict(tab_batch_sizes, 32)
    deg_batch_sizes = _optional_default_dict(deg_batch_sizes, 32)
    os.makedirs(save_db_to, exist_ok=True)

    for name, table in real_db.tables():
        table = Table.load(table)
        table_temp_cache = os.path.join(temp_cache, name)
        print('loaded table', name, flush=True)
        if os.path.exists(os.path.join(save_db_to, f'{name}.pkl')):
            print('get from generated', flush=True)
            gen_table = Table.load(os.path.join(save_db_to, f'{name}.pkl'))
        elif table.ttype == 'base':
            gen_table = table.shallow_copy()
            gen_table.update_temp_cache(table_temp_cache)
        elif table.is_independent():
            gen_table = _generate_independent_table(tab_models[name], table, scaling[name], tab_batch_sizes[name],
                                                    table_temp_cache)
        else:
            gen_table = _generate_dependent_table(tab_models[name], deg_models[name], table, scaling[name],
                                                  tab_batch_sizes[name], deg_batch_sizes[name], syn_db,
                                                  table_temp_cache)
        print('!!! assigned table new', name, flush=True)
        syn_db[name] = gen_table
        if name == 'personal_data':
            data = gen_table.data()
            print('%%% personal data??')
            print(set(range(data['student_token'].min(), data['student_token'].max())) - {*data['student_token']})
        print('done', flush=True)
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
    syn_table = SyntheticTable.from_real(table, temp_cache)
    syn_db.save_dummy(table.name, syn_table)
    # syn_db[table.name] = syn_table
    # known = syn_db.degree_known_for(table.name)
    known_tensors, augmented = [], []
    os.makedirs(os.path.join(temp_cache, 'deg_temp'), exist_ok=True)
    deg_cnt = 0
    while not syn_db.deg_finished(table.name):
        if os.path.exists(os.path.join(temp_cache, 'deg_temp', f'set{deg_cnt}.pt')):
            print('load from temp', flush=True)
            known_tab, aug_tab = torch.load(os.path.join(temp_cache, 'deg_temp', f'set{deg_cnt}.pt'))
        else:
            print('re join', flush=True)
            known, expected_size = syn_db.degree_known_for(table.name)
            print('gotten deg known', expected_size, known.shape, known.isnan().sum(), flush=True)
            if known.isnan().sum() > 0:
                raise ValueError()
            deg_tensor = deg_trainer.inference(known, deg_batch_size).output[:, -deg_trainer.unknown_dim:].cpu()
            # deg_tensors.append(deg_tensor)
            print('get inference', flush=True)
            degrees = syn_table.inverse_transform_degrees(deg_tensor, scale, expected_size)
            print('predicted degrees', table.name, len(degrees), degrees.sum(), degrees.describe())
            syn_table.assign_degrees(degrees)
            print('assigned deg', flush=True)
            known_tab, _, _ = syn_table.ptg_data()
            aug_tab = syn_table.data('augmented')
            print('degree assigned??', known_tab.shape, known_tab.isnan().sum(), flush=True)
            if known_tab.isnan().sum() > 0 and known_tab.shape[0] > 0:
                raise ValueError()
            torch.save((known_tab, aug_tab), os.path.join(temp_cache, 'deg_temp', f'set{deg_cnt}.pt'))
        print('new known tab', known_tab.shape)
        known_tensors.append(known_tab)
        augmented.append(aug_tab)
        deg_cnt += 1
    known_tab = torch.cat(known_tensors)
    augmented = pd.concat(augmented)
    syn_table.update_augmented(augmented)
    print('complete augmented table')
    print(augmented.sample(10))
    print('done getting degrees', known_tab.shape, flush=True)

    output = tab_trainer.inference(known_tab, tab_batch_size).output[:, -tab_trainer.unknown_dim:].cpu()
    print('inference has nan?', output.shape, known_tab.isnan().sum().sum(), output.isnan().sum().sum(), flush=True)
    recovered = syn_table.inverse_transform(output, replace_content=True)
    key_cols = [col for col in recovered.columns if 'student_token' in col or 'module_code' in col]
    print('??? recovered', recovered[key_cols].shape, recovered[key_cols].isnull().sum())
    # print('get inversely transformed', recovered[recovered[key_cols].isnull().sum() == 0].head())
    if len(recovered) > 0 and recovered[key_cols].isnull().sum().sum() > 0:
        raise ValueError()
    print('recovered')
    print(recovered.sample(10))
    syn_table.update_deg_and_aug()
    return syn_table
