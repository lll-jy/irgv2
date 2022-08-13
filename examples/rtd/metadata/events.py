"""Events-related tables."""

from typing import Dict, Any

import pandas as pd

from irg.schema import Table


def events(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['eventid', 'country']
    num_cat_cols = ['know_m', 'know_d', 'extended', 'specificity', 'vicinity', 'crit1', 'crit2', 'crit3',
                    'doubtterr', 'alternative', 'multiple', 'success', 'suicide', 'individual',
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
                'parent': 'events'
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
            'parent_columns': ['country_txt']
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
        }]
    }


def attack_type(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['eventid']
    num_cat_cols = ['attacktype']
    attributes = Table.learn_meta(src, id_cols, num_cat_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'foreign_keys': [{
            'columns': ['eventid'],
            'parent': 'events',
        }]
    }


def target(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['eventid']
    num_cat_cols = ['targtype', 'targsubtype', 'natlty']
    attributes = Table.learn_meta(src, id_cols, num_cat_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'determinants': [
            ['targtype', 'targtype_txt'],
            ['targsubtype', 'targsubtype_txt'],
            ['natlty_txt', 'natlty']
        ],
        'foreign_keys': [{
            'columns': ['eventid'],
            'parent': 'events',
        }, {
            'columns': ['natlty_txt'],
            'parent': 'country',
            'parent_columns': ['country_txt']
        }]
    }


def group(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['eventid']
    num_cat_cols = ['guncertain']
    attributes = Table.learn_meta(src, id_cols, num_cat_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'foreign_keys': [{
            'columns': ['eventid'],
            'parent': 'events',
        }]
    }


def claim(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['eventid']
    num_cat_cols = ['claim', 'claimmode']
    attributes = Table.learn_meta(src, id_cols, num_cat_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'determinants': [
            ['claimmode', 'claimmode_txt'],
        ],
        'foreign_keys': [{
            'columns': ['eventid'],
            'parent': 'events',
        }]
    }


def weapon(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['eventid']
    num_cat_cols = ['weaptype', 'weapsubtype']
    attributes = Table.learn_meta(src, id_cols, num_cat_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'determinants': [
            ['weaptype', 'weaptype_txt'],
            ['weapsubtype', 'weapsubtype_txt']
        ],
        'foreign_keys': [{
            'columns': ['eventid'],
            'parent': 'events',
        }]
    }


def related(src: pd.DataFrame) -> Dict[str, Any]:
    id_cols = ['eventid', 'related']
    attributes = Table.learn_meta(src, id_cols)
    return {
        'id_cols': id_cols,
        'attributes': attributes,
        'primary_keys': id_cols,
        'foreign_keys': [{
            'columns': ['eventid'],
            'parent': 'events',
        }, {
            'columns': ['related'],
            'parent': 'events',
            'parent_columns': ['eventid']
        }]
    }
