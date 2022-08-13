"""Events-related tables."""

from typing import Dict, Any

import pandas as pd

from irg.schema import Table


def events(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['eventid']
    num_cat_cols = ['know_m', 'know_d', 'extended', 'specificity', 'vicinity', 'crit1', 'crit2', 'crit3',
                    'doubtterr', 'alternative', 'multiple', 'success', 'suicide', 'individual', 'claimed',
                    'claimmode', 'compclaim', ]
    attributes = Table.learn_meta(src, id_cols, num_cat_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'determinants': [
            ['alternative', 'alternative_txt'],
        ],
        'primary_keys': id_cols,
        'foreign_keys': [
            {
                'columns': ['country', 'provstate', 'city'],
                'parent': 'city',
            }
        ]
    }


def life_damage(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['eventid']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'primary_keys': id_cols,
        'foreign_keys': [
            {
                'columns': ['eventid'],
                'parent': 'events',
            }
        ]
    }


def eco_damage(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['eventid']
    num_cat_cols = ['property', 'propextent']
    attributes = Table.learn_meta(src, id_cols, num_cat_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'determinants': [
            ['propextent', 'propextent_txt']
        ],
        'primary_keys': id_cols,
        'foreign_keys': [
            {
                'columns': ['eventid'],
                'parent': 'events',
            }
        ]
    }


def hostkid(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['eventid']
    num_cat_cols = ['ishostkid', 'ransom', 'hostkidoutcome']
    attributes = Table.learn_meta(src, id_cols, num_cat_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'determinants': [
            ['hostkidoutcome', 'hostkidoutcome_txt']
        ],
        'primary_keys': id_cols,
        'foreign_keys': [{
            'columns': ['eventid'],
            'parent': 'events',
        }, {
            'columns': ['kidhijcountry'],
            'parent': 'country',
            'parent_columns': ['country']
        }]
    }


def info_int(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['eventid']
    num_cat_cols = ['INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY']
    attributes = Table.learn_meta(src, id_cols, num_cat_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'primary_keys': id_cols,
        'foreign_keys': [{
            'columns': ['eventid'],
            'parent': 'events',
        },]
    }

