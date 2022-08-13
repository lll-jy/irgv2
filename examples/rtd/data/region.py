"""Region-related tables."""

import pandas as pd


def country(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Processed table**:

    Contains only regions and country.
    """
    return src[['country', 'country_txt', 'region', 'region_txt']].drop_duplicates().reset_index(drop=True)
