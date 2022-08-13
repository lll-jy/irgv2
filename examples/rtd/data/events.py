"""Events-related tables."""

from datetime import datetime, timedelta

from dateutil import parser
import numpy as np
import pandas as pd


def _parse_date_range_start(row: pd.Series):
    item = row['approxdate']
    if pd.isnull(item) or not item:
        return np.nan
    item = item.replace('Late', '25-28')
    item = item.replace('Early', '1-5')
    item = item.replace('First week of', '1-7')
    try:
        return parser.parse(item)
    except:
        if item.startswith('On or'):
            result = parser.parse(item[13:])
            result -= timedelta(days=1)
            return result
        if item == 'Yes':
            return datetime(year=row['iyear'],
                            month=max(1, row['imonth']),
                            day=max(1, row['iday']))
        if item.startswith('Within'):
            return datetime(year=row['iyear'], month=max(1, row['imonth']), day=1)
        if item.startswith('Overnight'):
            return datetime(year=2002, month=2, day=27, hour=21)
        if '-' in item:
            if item.count('-') == 2:
                left, mid, right = item.split('-')
                right = ' '.join([mid, right])
            else:
                left, right = item.split('-')
        else:
            left, right = item.split('or')
        right = right.replace(',', ', ')
        right = right.split()[1:]
        while right:
            try:
                result = parser.parse(left + ' ' + ' '.join(right))
                return result
            except:
                right = right[1:]


def _parse_date_range_end(row: pd.Series):
    item = row['approxdate']
    if pd.isnull(item) or not item:
        return np.nan
    item = item.replace('Late', '25-28')
    item = item.replace('Early', '1-5')
    item = item.replace('First week of', '1-7')
    try:
        return parser.parse(item)
    except:
        if item.startswith('On or'):
            result = parser.parse(item[13:])
            if item.startswith('On or around'):
                result += timedelta(days=1)
            return result
        if item == 'Yes':
            return datetime(year=row['iyear'],
                            month=max(1, row['imonth']),
                            day=max(1, row['iday']))
        if item.startswith('Within'):
            return datetime(year=row['iyear'], month=max(1, row['imonth']), day=10)
        if item.startswith('Overnight'):
            return datetime(year=2002, month=2, day=28, hour=8)
        if '-' in item:
            if item.count('-') == 2:
                left, mid, right = item.split('-')
                right = ' '.join([mid, right])
            else:
                left, right = item.split('-')
        else:
            left, right = item.split('or')
        try:
            return parser.parse(right)
        except:
            left = left.replace(',', ', ')
            left = left.split()[:-1]
            while left:
                try:
                    result = parser.parse(' '.join(left) + ' ' + right)
                    return result
                except:
                    left = left[:-1]


# TODO: Natural Language: 'location', 'summary', 'weapdetail', 'propcomment', 'ransomnote', 'addnotes', 'scite1',
#  'scite2', 'scite3'
def events(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Processed table**:

    Basic information, including location and timing of the incidents, general outcome, participant information, etc.
    """
    result = src[['eventid', 'country', 'provstate', 'city', 'extended', 'resolution', 'latitude', 'longitude',
                  'specificity', 'vicinity', 'crit1', 'crit2', 'crit3', 'doubtterr', 'alternative', 'alternative_txt',
                  'multiple', 'success', 'suicide', 'individual', 'nperps', 'nperpcap', 'claimed', 'compclaim',
                  'nhours', 'ndays', 'divert', 'dbsource']]
    result['idate'] = src.apply(lambda row: datetime(
        year=row['iyear'],
        month=max(1, row['imonth']),
        day=max(1, row['iday'])), axis=1)
    result['know_m'] = (src['imonth'] == 1).astype(int)
    result['know_d'] = (src['iday'] == 1).astype(int)
    result['approx_start'] = src.apply(_parse_date_range_start, axis=1)
    result['approx_end'] = src.apply(_parse_date_range_end, axis=1)
    result['resolution'] = pd.to_datetime(result['resolution'])
    for col in ['specificity', 'doubtterr', 'alternative', 'multiple', 'claimed', 'compclaim']:
        result[col] = result[col].astype('Int32')
    return result


def life_damage(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Processed table**:

    Damages on human beings per event. That is, how many are killed, wounded, etc.
    """
    src = src[['eventid', 'nkill', 'nkillus', 'nkillter', 'nwound', 'nwoundus', 'nwoundte']]
    return src


def eco_damage(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Processed table**:

    Economic damages per event.
    """
    src = src[['eventid', 'property', 'propextent', 'propextent_txt', 'propvalue']]
    src['propextent'] = src['propextent'].astype('Int32')
    return src


def _process_country(c):
    if pd.isnull(c):
        return np.nan
    if c in {'Arrested before boarding',
             'Trinidad-This; Aruba-12 hrs; Curacas',
             'Unknown',
             'no (police foiled operation)',
             'not stated',
             'unknown'}:
        return np.nan
    kw_map = {
        'Moscow': 'Russia',
        'Paris': 'France',
        'Pakistan': 'Pakistan',
        'Changi': 'Singapore',
        'Tehran': 'Iran',
        'Ethiopia': 'Ethiopia',
        'Amalfi': 'Italy',
        'USA': 'United States',
        'Baghdad': 'Iraq',
        'Bandar Abbas': 'Iran',
        'Beirut': 'Lebanon',
        'Teheran': 'Iran',
        'Cali': 'Colombia',
        'Brussels': 'Belgium',
        'Cologne': 'Germany',
        'Congo': 'Republic of the Congo',
        'Costa Rica': 'Costa Rica',
        'Cuba': 'Cuba',
        'Corsica': 'France',
        'Cucuta': 'Colombia',
        'Turkley': 'Turkey',
        'Djibouti': 'Djibouti',
        'Durban': 'South Africa',
        'Russia': 'Russia',
        'Britain': 'United Kingdom',
        'England': 'United Kingdom',
        'Paramaribo': 'Suriname',
        'Geneva': 'Switzerland',
        'Greece': 'Greece',
        'Colombia': 'Colombia',
        'Panama': 'Panama',
        'Higuerote': 'Venezuela',
        'Ho Chi Minh City': 'Vietnam',
        'Kasese': 'Uganda',
        'Lima': 'Peru',
        'Israel': 'Israel',
        'Madrid': 'Spain',
        'Marseilles': 'France',
        'Medellin': 'Colombia',
        'Northern Ireland': 'Ireland',
        'Saudi': 'Saudi Arabia',
        'Palestine': 'Israel',
        'Port Sudan': 'Sudan',
        'Prague': 'Czech Republic',
        'Puerto Rico': 'Dominican Republic',
        'Salonika': 'Greece',
        'Stockholm': 'Sweden',
        'Tegucigalpa': 'Honduras',
        'Tel Aviv': 'Israel',
        'Philippines': 'Philippines',
        'Neiva': 'Colombia'
    }
    for alias, country_name in kw_map.items():
        if alias in c:
            return country_name
    return c.split(', ')[-1]


def hostkid(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Processed table**:

    Host kidnapping per event, including ransom information.
    Some processing on the `kidhijcountry` is done such that all countries can be found in declared regions.
    Some cells in this column contain only city or region information, which are mapped to corresponding countries.
    In some rare cases, some countries are not recognized, in which case the nearest country's name is used.
    """
    src = src[['eventid', 'ishostkid', 'nhostkid', 'kidhijcountry', 'ransom', 'ransomamt', 'ransomamtus',
               'ransompaid', 'ransompaidus', 'hostkidoutcome', 'hostkidoutcome_txt', 'nreleased']]
    for col in ['ishostkid', 'ransom', 'hostkidoutcome']:
        src[col] = src[col].astype('Int32')
    src['kidhijcountry'] = src['kidhijcountry'].apply(_process_country)
    return src


def info_int(src: pd.DataFrame) -> pd.DataFrame:
    """
    **Processed table**:

    `INT_` information per event.
    """
    return src[['eventid', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY']]
