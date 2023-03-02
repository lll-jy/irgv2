"""TIme-series connection tables."""

from typing import Dict, Any

import pandas as pd

from irg.schema import Table


def wifi(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['student_token']
    embed = ['aplocation', 'aplocation_mapping', 'apname']
    attributes = Table.learn_meta(src, id_cols, force_embed=embed)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'primary_keys': ['student_token', 'sessionstarttime'],
        'ttype': 'series',
        'series_id': 'sessionstarttime',
        'base_columns': ['student_token', 'day'],
        'foreign_keys': [{
            'columns': ['student_token'],
            'parent': 'personal_data'
        }]
    }


def luminus(src: pd.DataFrame) -> Dict[str, Any]:
    attributes = Table.learn_meta(src, ['student_token'])
    return {
        'id_cols': ['student_token'],
        'attributes': attributes,
        'primary_keys': ['student_token', 'recorddate_r'],
        'ttype': 'series',
        'series_id': 'recorddate_r',
        'base_columns': ['student_token'],
        'foreign_keys': [{
            'columns': ['student_token'],
            'parent': 'personal_data'
        }]
    }
