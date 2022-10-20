"""
Census dataset data pre-processing.

The dataset is retrieved from http://archive.ics.uci.edu/ml/datasets/Census-Income+(KDD).
"""

import os

import pandas as pd

from .processor import CensusProcessor, CENSUS_PROCESSORS, CENSUS_META_CONSTRUCTORS, CENSUS_PROCESS_NAME_MAP


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
    names = ['age', 'wc', 'ind', 'occ', 'edu', 'wph', 'in_edu', 'marital', 'm_ind', 'm_occ', 'race', 'his', 'sex',
             'lu', 'unemploy', 'ftpt', 'gain', 'loss', 'stock', 'tax', 'region', 'state', 'household_family',
             'household', 'change_msa', 'change_reg', 'move_reg', 'yr_ago', 'mig_prev', 'n_work', 'n_child',
             'country_father', 'country_mother', 'country', 'citizenship', 'own_business', 'veteran_fill',
             'veteran_benefit', 'wkpy', 'yr', 'label']
    cont_names = ['age', 'wph', 'gain', 'loss', 'stock', 'n_work', 'wkpy']
    train = pd.read_csv('census-income.data', names=names, na_values='?', sep=', ')
    test = pd.read_csv('census-income.test', names=names, na_values='?', sep=', ')
    need_to_cast = []
    for n in names:
        if n not in cont_names:
            try:
                train[n].astype('float32')
                test[n].astype('float32')
                need_to_cast.append(n)
            except ValueError:
                pass
    for c in need_to_cast:
        train.loc[:, c] = train[c].apply(lambda x: f'c{x}')
        test.loc[:, c] = test[c].apply(lambda x: f'c{x}')
    train.to_csv(os.path.join(data_dir, 'census.train.csv'), index=False)
    test.to_csv(os.path.join(data_dir, 'census.test.csv'), index=False)
    os.remove('census-income.data')
    os.remove('census-income.test')


if __name__ == '__main__':
    download('../data.nosync/census')
