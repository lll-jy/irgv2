"""
CLI tool to process data.

Run this script with `python3 PATH/TO/THIS/FILE -h` for CLI argument helpers.
"""

from argparse import ArgumentParser, Namespace
import os

from examples import DATABASE_PROCESSORS, DatabaseProcessor


def _parse_args() -> Namespace:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='target')

    table_parser = subparsers.add_parser('table')
    table_parser.add_argument('mode', type=str, help='Content to process.', choices=['data', 'meta'])
    table_parser.add_argument('database_name', type=str, help='Name of database to process.',
                              choices=['alset', 'rtd', 'adults', 'covtype'])
    table_parser.add_argument('table_name', type=str, help='Name of the table to process.')
    table_parser.add_argument('--src', type=str, required=True, help='Source data file.')
    table_parser.add_argument('--out', type=str, required=True,
                              help='Output file path or directory. If directory is given, the file is saved under the '
                                   'directory with file name the same as table_name.')

    db_parser = subparsers.add_parser('database')
    db_parser.add_argument('database_name', type=str, help='Name of database to process.',
                           choices=['alset', 'rtd', 'adults', 'covtype'])
    db_parser.add_argument('--src_data_dir', type=str, default=None, help='Path to original data.')
    db_parser.add_argument('--data_dir', type=str, default=None, help='Path to directory holding processed data.')
    db_parser.add_argument('--meta_dir', type=str, required=True, help='Path to directory holding metadata files.')
    db_parser.add_argument('--tables', type=str, nargs='*', default=[], help='Tables to include.')
    db_parser.add_argument('--out', type=str, required=True, help='Output file path of database.')
    db_parser.add_argument('--redo_meta', default=False, action='store_true',
                           help='Whether to reprocess metadata if file already exists.')
    db_parser.add_argument('--redo_data', default=False, action='store_true',
                           help='Whether to reprocess table data if file already exists.')
    db_parser.add_argument('--sample', default=None, type=int, help='Generate a smaller version of the database.')
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.target == 'table':
        processor: DatabaseProcessor = DATABASE_PROCESSORS[args.database_name].create_dummy()
        if args.mode == 'data':
            processor.process_table_data(args.table_name, args.src, args.out)
        else:
            processor.construct_table_metadata(args.table_name, args.src, args.out)
    else:
        processor: DatabaseProcessor = DATABASE_PROCESSORS[args.database_name](args.src_data_dir, args.data_dir,
                                                                               args.meta_dir, args.out, args.tables)
        processor.process_database(args.redo_meta, args.redo_data)
        if args.sample is not None:
            processor.postprocess(os.path.join(args.data_dir, 'samples'), args.sample)


if __name__ == '__main__':
    main()
