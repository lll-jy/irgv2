"""Gym-related tables."""

from typing import Dict, Any

import pandas as pd

from irg.schema import Table


def uci_gym(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['student_token']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'primary_keys': ['student_token', 'date', 'check_in_time'],
        'foreign_keys': [{
            'columns': ['student_token'],
            'parent': 'personal_data'
        }]
    }
