"""
Census dataset data pre-processing.

The dataset is retrieved from http://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD).
"""

import os

import pandas as pd

from .processor import CensusProcessor, CENSUS_PROCESSORS, CENSUS_META_CONSTRUCTORS, CENSUS_PROCESS_NAME_MAP


def _process(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[:, 'education'] = df['education'].replace({
        'Children': 0,
        'Less than 1st grade': 1,
        '1st 2nd 3rd or 4th grade': 2,
        '5th or 6th grade': 3,
        '7th and 8th grade': 4,
        '9th grade': 5,
        '10th grade': 6,
        '11th grade': 7,
        '12th grade no diploma': 8,
        'High school graduate': 9,
        'Some college but no degree': 10,
        'Associates degree-occup /vocational': 11,
        'Associates degree-academic program': 12,
        'Bachelors degree(BA AB BS)': 13,
        'Masters degree(MA MS MEng MEd MSW MBA)': 14,
        'Prof school degree (MD DDS DVM LLB JD)': 15,
        'Doctorate degree(PhD EdD)': 16
    }).astype('int32')
    for c in ['industry', 'occupation', 'own_business', 'veteran_benefit', 'year']:
        df.loc[:, c] = df[c].apply(lambda x: f'c{x}').astype('O')
    for c in ['age', 'wph', 'cap_gain', 'cap_loss', 'div_stock', 'instance_weight', 'n_work', 'wk_per_year']:
        df.loc[:, c] = df[c].astype('float32')
    return df


def download(data_dir: str):
    """
    Download Census dataset.

    **Args**:

    - `data_dir` (`str`): Target directory to save the data.
    """
    os.makedirs(data_dir, exist_ok=True)
    os.system('wget http://kdd.ics.uci.edu/databases/census-income/census-income.data.gz')
    os.system('wget http://kdd.ics.uci.edu/databases/census-income/census-income.test.gz')
    os.system('gunzip census-income.data.gz')
    os.system('gunzip census-income.test.gz')
    names = [
        'age', 'class_of_worker', 'industry', 'occupation', 'education', 'wph', 'enrol_last_wk', 'marital',
        'major_industry', 'major_occupation', 'race', 'hisp', 'sex', 'member_labor_union', 'reason_unemploy', 'ftpt',
        'cap_gain', 'cap_loss', 'div_stock', 'taxfiler', 'region', 'state', 'householder', 'householder_summary',
        'instance_weight', 'change_msa', 'change_reg', 'move_reg', 'yr_ago', 'mig_prev_sunbelt', 'n_work',
        'parent_present', 'country_father', 'country_mother', 'country', 'citizenship', 'own_business',
        'veteran_fill', 'veteran_benefit', 'wk_per_year', 'year', 'label'
    ]
    train = pd.read_csv('census-income.data', names=names, na_values='?', sep=', ')
    test = pd.read_csv('census-income.test', names=names, na_values='?', sep=', ')
    train = _process(train)
    test = _process(test)
    train.to_csv(os.path.join(data_dir, 'census.train.csv'), index=False)
    test.to_csv(os.path.join(data_dir, 'census.test.csv'), index=False)
    os.remove('census-income.data')
    os.remove('census-income.test')


if __name__ == '__main__':
    download('../data.nosync/census')
