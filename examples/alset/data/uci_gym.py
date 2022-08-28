"""Gym-related tables."""

import pandas as pd


def uci_gym(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Raw table**:

    UCI Gym data.

    **Processed table**:

    Students' gym check-in and check-out records.
    """
    return src.astype({
        'date': 'datetime64[ns]',
        'check_in_time': 'datetime64[ns]',
        'check_out_time': 'datetime64[ns]'
    })

