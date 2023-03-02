"""Time-series data tables."""
import pandas as pd


def wifi(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Processed table**:

    WiFi connection data.
    """
    src['sessionlast'] = src['sessionendtime'].astype('datetime64[ns]') - \
                         src['sessionstarttime'].astype('datetime64[ns]')
    src = src.drop(columns=['ipaddress_token', 'mac_token', 'sessionendtime'])
    datetime_cols = ['sessionstarttime', 'day']
    src = src.astype({d: 'datetime64[ns]' for d in datetime_cols})
    return src


def luminus(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Processed table**:

    LumiNUS actions.
    """
    return src.astype({'recorddate_r': 'datetime64[ns]'})
