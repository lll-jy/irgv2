"""
CLI tool to process data.

Run this script with `python3 PATH/TO/THIS/FILE -h` for CLI argument helpers.
"""

from argparse import ArgumentParser, Namespace
import os
from typing import Dict
import json

import pandas as pd

from examples import PROCESSORS, META_CONSTRUCTORS

_ENCODINGS: Dict[str, str] = {
    'alset': None,
    'rtd': 'ISO-8859-1'
}


def _parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('mode', type=str, help='Content to process.', choices=['data', 'meta'])
    parser.add_argument('database_name', type=str, help='Name of database to process.',
                        choices=['alset', 'rtd'])
    parser.add_argument('table_name', type=str, help='Name of the table to process.')
    parser.add_argument('--src', type=str, required=True, help='Source data file.')
    parser.add_argument('--out', type=str, required=True,
                        help='Output file path or directory. If directory is given, the file is saved under the '
                             'directory with file name the same as table_name.')
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.mode == 'data':
        src = pd.read_csv(args.src, encoding=_ENCODINGS[args.database_name])
        out = os.path.join(args.out, f'{args.table_name}.pkl') if os.path.isdir(args.out) else args.out
        processed: pd.DataFrame = PROCESSORS[args.database_name][args.table_name](src)
        processed.to_pickle(out)
    else:
        src = pd.read_pickle(args.src)
        out = os.path.join(args.out, f'{args.table_name}.json') if os.path.isdir(args.out) else args.out
        constructed: Dict = META_CONSTRUCTORS[args.database_name][args.table_name](src)
        with open(out, 'w') as f:
            json.dump(constructed, f, indent=2)


if __name__ == '__main__':
    main()
