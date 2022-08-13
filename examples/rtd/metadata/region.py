"""Region-related tables."""

from typing import Dict, Any

import pandas as pd

from irg.schema import Table


def country(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['country']
    attributes = Table.learn_meta(src, id_cols, ['region'])
    return {
        'id_cols': id_cols,
        'ttype': 'base',
        'attributes': attributes,
        'determinants': [
            ["country", "country_txt"],
            ["region", "region_txt"]
        ],
        'primary_keys': id_cols
    }


def provstate(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['country']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'ttype': 'base',
        'attributes': attributes,
        'primary_keys': ['country', 'provstate'],
        'foreign_keys': [
            {
                'columns': ['country'],
                'parent': 'country',
            }
        ]
    }


def city(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['country']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'ttype': 'base',
        'attributes': attributes,
        'primary_keys': ['country', 'provstate', 'city'],
        'foreign_keys': [
            {
                'columns': ['country', 'provstate'],
                'parent': 'provstate',
            }
        ]
    }
