import datetime
from typing import Dict, List, Optional
from types import FunctionType
import shutil
import os
import json
import pandas as pd
from collections import deque
from sdv import Metadata
from sdv.relational import HMA1
from sdv.tabular import CTGAN



def origin_data(pro_path: str,meta_path: str):
    #load process_data
    file = os.listdir(pro_path)
    process_datas = dict()
    for f in file:
        table = f.split('.')
        if len(table) > 1 and table[1] == "pkl":
            process_datas[table[0]] = pd.read_pickle(pro_path + f)

    # load meta_data
    file = os.listdir(meta_path)
    metadatas = dict()
    for f in file:
        table = f.split('.')[0]
        with open(meta_path + f, 'r') as data:
            metadatas[table] = json.load(data)

    original_joined = merge_table(process_datas, metadatas)
    no_nan_original_joined = construct_without_nan(original_joined)
    no_nan_original = {item[0]: construct_without_nan(item[1]) for item in process_datas.items()}
    tables_with_dummy_id = add_id_columns(no_nan_original)
    return {
        "process_datas" : process_datas,
        "metadatas" : metadatas,
        "original_joined" : original_joined,
        "no_nan_original_joined" : no_nan_original_joined,
        "no_nan_original" : no_nan_original,
        "tables_with_dummy_id" : tables_with_dummy_id
    }

def sdv (process_data: dict, metadatas: dict, tables_with_dummy_id: dict):
    sdv_metadata = Metadata()
    for key in process_data.keys():
        metadata = metadatas[key]
        data = process_data[key].copy()

        # check if exist 'parent_columns'
        for relationship in metadata['foreign_keys']:
            if 'parent_columns' in relationship.keys():
                data.rename(columns={relationship['columns'][0]: relationship['parent_columns'][0]}, inplace=True)
                # add_table
        sdv_metadata.add_table(name=key, data=data, primary_key=metadata['primary_keys'][0])

        # add_relationship
        for relationship in metadata['foreign_keys']:
            sdv_metadata.add_relationship(parent=relationship['parent'], child=key, foreign_key=relationship[
                'parent_columns'] if 'parent_columns' in relationship.keys() else relationship['columns'])

    sdv_model = HMA1(sdv_metadata)
    sdv_model.fit(tables_with_dummy_id)
    sdv_gemerated = sdv_model.sample()
    store_generated('sdv', sdv_gemerated)

def ctgan(no_nan_original_joined: pd.DataFrame):
    ctgan_joined_model = CTGAN()
    ctgan_joined_model.fit(no_nan_original_joined)
    ctgan_joined_generated_joined = ctgan_joined_model.sample(len(no_nan_original_joined))
    ctgan_joined_generated = decompose_ctgan_joined(ctgan_joined_generated_joined)
    store_generated('ctgan joined', ctgan_joined_generated)
    ctgan_joined_generated_joined.to_csv('generated/exp1/ctgan joined.csv', index=False)

def ctgan_separate(no_nan_original: pd.DataFrame):
    ctgan_separate_generated = {}
    for name, table in no_nan_original.items():
        ctgan_separate_model = CTGAN()
        ctgan_separate_model.fit(table)
        ctgan_separate_generated[name] = ctgan_separate_model.sample(len(table))
    store_generated('ctgan separate', ctgan_separate_generated)



def merge_table(tables: dict, metadata: dict):
    original_joined = tables["personal_data"]
    print("merge personal_data successfully")
    dq = deque(tables.keys())
    while dq:
        table_name = dq.popleft()
        print(table_name)
        if table_name != "personal_data":
            merge_success = False
            foreign_keys = metadata[table_name]["foreign_keys"] if "foreign_keys" in metadata[table_name].keys() else [{'columns':metadata[table_name]['primary_keys']}]
            for fk in foreign_keys:
                if set(fk["columns"]) < set(original_joined.columns):
                    cols_diff = list(tables[table_name].columns.difference(original_joined.columns))
                    cols_diff.extend(fk["columns"])
                    print(f"merge {table_name} start")
                    original_joined = original_joined.merge(tables[table_name][cols_diff], on = fk["columns"], how = 'outer')
                    merge_success = True
                    print(f"merge {table_name} successfully")
                    break
            if not merge_success:
                dq.append(table_name)
    return original_joined

def construct_without_nan(original: pd.DataFrame):
    no_nan_data = original.copy()
    for col, col_type in dict(no_nan_data.dtypes).items():
        if col == 'exchange_level':
            no_nan_data[col] = 0
            continue
        if str(col_type) == 'object':
            no_nan_data[col] = no_nan_data[col].fillna('nan!!')
            no_nan_data[col] = no_nan_data[col].apply(str)
        else:
            mean_val = no_nan_data[col].mean()
            if pd.isnull(mean_val):
                if str(col_type) == 'datetime[ns]':
                    mean_val = datetime.now()
                else:
                    mean_val = 0
            elif str(col_type).startswith('Int') or str(col_type).startswith('int'):
                mean_val = round(mean_val)
            no_nan_data[col] = no_nan_data[col].fillna(mean_val)
            if str(col_type) == 'Int64':
                no_nan_data = no_nan_data.astype({col: 'int32'})
    return no_nan_data

def add_id_columns(tables: dict) -> dict:
    new_tables = {}
    for name in tables:
        new_tables[name] = tables[name].copy()
        new_tables[name][f'{name}_id'] = new_tables[name].index

def store_generated(model_name: str, generated_tables: dict):
    for name in generated_tables:
        table = generated_tables[name]
        table.to_csv('generated/exp1/' + model_name + '_' + name + '.csv', index=False)
        table.to_pickle('generated/exp1/' + model_name + '_' + name + '.pkl')

def decompose_ctgan_joined(joined_table: pd.DataFrame,no_nan_original: pd.DataFrame) -> dict:
    column_names = {name:[*table.columns] for name, table in no_nan_original.items()}
    joined = {name:joined_table[cols] for name, cols in column_names.items()}
    for table_name, table in joined.items():
        joined[table_name] = joined[table_name].drop_duplicates()
    return joined