"""Time-series data tables."""
import pandas as pd


def wifi(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Processed table**:

    WiFi connection data.
    """
    src['sessionlast'] = src['sessionendtime'] - src['sessionstarttime']
    src = src.drop(columns=['ipaddress_token', 'mac_token', 'sessionendtime'])
    datetime_cols = ['sessionstarttime', 'day']
    src = src.astype({d: 'datetime64[ns]' for d in datetime_cols})
    return datetime_cols
