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
        'primary_keys': ['student_token'],
        'ttype': 'series',
    }
