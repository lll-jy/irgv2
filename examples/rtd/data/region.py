"""Region-related tables."""

import pandas as pd


def country(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Processed table**:

    Contains only regions and country.
    """
    return src[['country', 'country_txt', 'region', 'region_txt']].drop_duplicates().reset_index(drop=True)


def provstate(src: pd.DataFrame) -> pd.DataFrame:
    """
    ***Processed table**:

    Contains provstate and corresponding country.
    """
    return src[['country', 'provstate']]


def city(src: pd.DataFrame) -> pd.DataFrame:
    """
    ***Processed table**:

    Contains city, provstate and corresponding country.
    """
    src = src[['country', 'provstate', 'city']]
    src['city'] = src['city'].str.title()
    return src
