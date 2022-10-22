"""AirBnb tables."""

from typing import Dict, Any

import pandas as pd

from irg.schema import Table


def countries(src: pd.DataFrame) -> Dict[str, Any]:
    attributes = Table.learn_meta(src)
    return {
        'attributes': attributes,
        'primary_keys': ['country_destination'],
        'ttype': 'base'
    }


def age_gender(src: pd.DataFrame) -> Dict[str, Any]:
    attributes = Table.learn_meta(src)
    return {
        'attributes': attributes,
        'primary_keys': ['age_bucket', 'country_destination', 'gender'],
        'ttype': 'base',
        'foreign_keys': [{
            'columns': ['country_destination'],
            'parent': 'countries'
        }]
    }


def users(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['id']
    num_cat = ['signup_flow']
    attributes = Table.learn_meta(src, id_cols, num_cat)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'primary_keys': ['id'],
        'foreign_keys': [{
            'columns': ['country_destination'],
            'parent': 'countries'
        }]
    }


def sessions(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['user_id']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'foreign_keys': [{
            'columns': ['user_id'],
            'parent': 'users',
            'parent_columns': ['id']
        }]
    }
