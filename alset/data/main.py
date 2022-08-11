"""CLI tool to process data."""

from argparse import ArgumentParser, Namespace
import os
from typing import Dict
from types import FunctionType

import pandas as pd

from alset.data.sis import personal_data

_PROCESSORS: Dict[str, FunctionType] = {
    'personal_data': personal_data
}


def _parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('table_name', type=str, required=True, help='Name of the table to process.')
    parser.add_argument('--src', type=str, required=True, help='Source data file.')
    parser.add_argument('--out', type=str, required=True,
                        help='Output file path or directory. If directory is given, the file is saved under the '
                             'directory with file name the same as table_name.')
    return parser.parse_args()


def main():
    args = _parse_args()
    src = pd.read_csv(args.src)
    out = os.path.join(args.out, f'{args.table_name}.pkl') if os.path.isdir(args.out) else args.out
    processed: pd.DataFrame = _PROCESSORS[args.table_name](src)
    processed.to_pickle(out)


if __name__ == '__main__':
    main()
