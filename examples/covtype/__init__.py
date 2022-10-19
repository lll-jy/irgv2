"""
CovType dataset data pre-processing.

The dataset is retrieved from http://archive.ics.uci.edu/ml/datasets/covertype.
"""

import os
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .processor import CovtypeProcessor, COVTYPE_PROCESSORS, COVTYPE_META_CONSTRUCTORS, COVTYPE_PROCESS_NAME_MAP

__all__ = (
    'CovtypeProcessor',
    'COVTYPE_PROCESSORS',
    'COVTYPE_META_CONSTRUCTORS',
    'COVTYPE_PROCESS_NAME_MAP',
    'download'
)


def _split_train_test(data: pd.DataFrame, names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(data)
    train = pd.DataFrame(train, columns=names)
    test = pd.DataFrame(test, columns=names)
    return train, test


def download(data_dir: str):
    """
    Download CovType dataset.

    **Args**:

    - `data_dir` (`str`): Target directory to save the data.
    """
    os.makedirs(data_dir, exist_ok=True)
    os.system('wget http://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz')
    os.system('gunzip covtype.data.gz')
    names = ['elevation', 'aspect', 'slope', 'hd_hyd', 'vd_hyd', 'hd_rw', 'hill9', 'hill_n', 'hill3', 'hd_fp'] + \
            [f'w{i}' for i in range(4)] + [f'st_{i:02d}' for i in range(40)] + ['label']
    data = pd.read_csv('covtype.data', names=names)
    train, test = _split_train_test(data, names)
    train.to_csv(os.path.join(data_dir, 'covtype.train.csv'), index=False)
    test.to_csv(os.path.join(data_dir, 'covtype.test.csv'), index=False)
    os.remove('covtype.data')


if __name__ == '__main__':
    download('../examples/data/covtype')
