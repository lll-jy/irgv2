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
    subparsers = parser.add_subparsers(dest='target')

    table_parser = subparsers.add_parser('table')
    table_parser.add_argument('mode', type=str, help='Content to process.', choices=['data', 'meta'])
    table_parser.add_argument('database_name', type=str, help='Name of database to process.',
                              choices=['alset', 'rtd'])
    table_parser.add_argument('table_name', type=str, help='Name of the table to process.')
    table_parser.add_argument('--src', type=str, required=True, help='Source data file.')
    table_parser.add_argument('--out', type=str, required=True,
                              help='Output file path or directory. If directory is given, the file is saved under the '
                                   'directory with file name the same as table_name.')

    db_parser = subparsers.add_parser('database')
    db_parser.add_argument('database_name', type=str, help='Name of database to process.',
                           choices=['alset', 'rtd'])
    db_parser.add_argument('--src_data_dir', type=str, default=None, help='Path to original data.')
    db_parser.add_argument('--data_dir', type=str, default=None, help='Path to directory holding processed data.')
    db_parser.add_argument('--meta_dir', type=str, required=True, help='Path to directory holding metadata files.')
    db_parser.add_argument('--tables', type=str, nargs='*', default=[], help='Tables to include.')
    db_parser.add_argument('--out', type=str, required=True, help='Output file path of database.')
    db_parser.add_argument('--redo_meta', default=False, action='store_true',
                           help='Whether to reprocess metadata if file already exists.')
    db_parser.add_argument('--redo_data', default=False, action='store_true',
                           help='Whether to reprocess table data if file already exists.')
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.target == 'table':
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
    else:
        os.makedirs(args.meta_dir, exist_ok=True)
        if not args.tables:
            args.tables = [*PROCESSORS[args.database]]

        out_config = {}
        for table_name in args.tables:
            meta_path = os.path.join(args.meta_dir, f'{table_name}.json')
            if not os.path.exists(meta_path) or args.redo_meta:
                if args.data_dir is None:
                    raise ValueError('Need to access processed data, please provide data_dir.')
                if not os.path.exists(args.data_dir):
                    os.makedirs(args.data_dir)
                data_path = os.path.join(args.data_dir, f'{table_name}.pkl')
                if not os.path.exists(data_path) or args.redo_data:
                    if args.src_data_dir is None:
                        raise ValueError('Need to access original data, please provide src_data_dir.')
                    src_data_path = os.path.join(args.src_data_path, f'{table_name}.csv')
                    src = pd.read_csv(src_data_path, encoding=_ENCODINGS[args.database_name])
                    processed: pd.DataFrame = PROCESSORS[args.database_name][args.table_name](src)
                    processed.to_pickle(data_path)
                processed = pd.read_pickle(args.data_path)
                constructed: Dict = META_CONSTRUCTORS[args.database_name][args.table_name](processed)
                with open(meta_path, 'w') as f:
                    json.dump(constructed, f, indent=2)
            with open(meta_path, 'r') as f:
                loaded = json.load(f)
                loaded['format'] = 'pickle'
                out_config[table_name] = loaded
        with open(args.out, 'w') as f:
            json.dump(out_config, f)


if __name__ == '__main__':
    main()
