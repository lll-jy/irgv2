"""AirBbn dataset processors."""

from datetime import datetime

import pandas as pd


def countries(src: pd.DataFrame) -> pd.DataFrame:
    return src.copy()


def age_gender(src: pd.DataFrame) -> pd.DataFrame:
    src = src.copy()
    src.loc[:, 'year'] = src['year'].apply(lambda x: datetime(year=x, month=1, day=1))
    return src


def users(src: pd.DataFrame) -> pd.DataFrame:
    return src.astype({
        'date_account_created': 'datetime64[ns]',
        'timestamp_first_active': 'datetime64[ns]',
        'date_first_booking': 'datetime64[ns]'
    })


def sessions(src: pd.DataFrame) -> pd.DataFrame:
    return src.copy()


