"""
Adults dataset data pre-processing.

The dataset is retrieved from https://archive.ics.uci.edu/ml/datasets/Adult.
"""

import os

import numpy as np
import pandas as pd

from .processor import AdultsProcessor, ADULTS_PROCESSORS, ADULTS_META_CONSTRUCTORS, ADULTS_PROCESS_NAME_MAP


__all__ = (
    'AdultsProcessor',
    'ADULTS_PROCESSORS',
    'ADULTS_META_CONSTRUCTORS',
    'ADULTS_PROCESS_NAME_MAP',
    'download'
)


def download(data_dir: str):
    """
    Download Adult dataset.

    **Args**:

    - `data_dir` (`str`): Target directory to save the data.
    """
    os.makedirs(data_dir, exist_ok=True)
    os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data')
    os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test')
    names = ['age', 'workclass', 'fnlwgt', 'education', 'edu_num', 'marital', 'occupation', 'relation', 'race',
             'sex', 'cap_gain', 'cap_loss', 'hpw', 'native', 'label']
    train = pd.read_csv('adult.data', names=names)
    test = pd.read_csv('adult.test', names=names, skiprows=[0])
    train = train.replace({'?': np.nan})
    test = test.replace({'?': np.nan})
    train.to_csv(os.path.join(data_dir, f'adults.train.csv'), index=False)
    test.to_csv(os.path.join(data_dir, f'adults.test.csv'), index=False)
    os.remove('adult.data')
    os.remove('adult.test')


if __name__ == '__main__':
    download('../examples/data/adults')
