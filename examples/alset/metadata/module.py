"""Module-related tables."""

from typing import Dict, Any

import pandas as pd

from irg.schema import Table


def module_offer(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['module_code']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'primary_keys': ['module_code'],
        'ttype': 'base'
    }


def module_enrolment(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['module_code', 'student_token']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'primary_keys': ['module_code', 'student_token', 'tyear', 'tsem'],
        'foreign_keys': [{
            'columns': ['student_token'],
            'parent': 'personal_data'
        }, {
            'columns': ['module_code'],
            'parent': 'module_offer',
        }]
    }
