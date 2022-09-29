"""Main runner script."""

from argparse import ArgumentParser, Namespace
from collections import defaultdict
import json
import pickle
import os
from typing import Optional, DefaultDict, Any, List
import random
import logging

import torch
import numpy as np

from irg.utils.dist import init_process_group
from irg import engine

_LOGGER = logging.getLogger()


def _parse_distributed_args(parser: ArgumentParser):
    group = parser.add_argument_group('Distributed')
    group.add_argument('--distributed', default=False, action='store_true',
                       help='Whether this is distributed training.')


def _parse_database_args(parser: ArgumentParser):
    group = parser.add_argument_group('Database')
    group.add_argument('--db_config_path', default='config.json', type=str,
                       help='File path that content of this database schema is saved.')
    group.add_argument('--engine', default='json', type=str, help='File format of database',
                       choices=['json', 'pickle', 'yaml', 'torch'])
    group.add_argument('--data_dir', default='.', type=str, help='Data directory with data content.')
    group.add_argument('--mtype', default='affecting', type=str, help='Augmentation mechanism of database.',
                       choices=['unrelated', 'parent-child', 'ancestor-descendant', 'affecting'])
    group.add_argument('--db_dir_path', default=None, type=str, help='Directory path of the real database.')
    group.add_argument('--aug_resume', default=False, action='store_true', help='Whether to resume from already '
                                                                                'augmented database if provided.')


def _parse_train_args(parser: ArgumentParser):
    group = parser.add_argument_group('Train')
    group.add_argument('--skip_train', default=True, action='store_false', dest='do_train',
                       help='Whether to skip training process.')

    def _add_trainer_args(short: str, descr: str):
        trainer_group = group.add_argument_group(f'{descr} model trainer constructors')
        trainer_group.add_argument(f'--default_{short}_trainer_args', default=None, type=str,
                                   help='Default arguments saved as JSON file.')
        trainer_group.add_argument(f'--default_{short}_trainer_trainer_type', default=None, type=str,
                                   choices=['CTGAN', 'TVAE', 'MLP'])
        trainer_group.add_argument(f'--default_{short}_trainer_distributed', default=None, type=bool)
        trainer_group.add_argument(f'--default_{short}_trainer_autocast', default=None, type=bool)
        trainer_group.add_argument(f'--default_{short}_trainer_log_dir', default=None, type=str)
        trainer_group.add_argument(f'--default_{short}_trainer_ckpt_dir', default=None, type=str)
        trainer_group.add_argument(f'--{short}_trainer_args', type=str, help='Arguments saved as JSON files as a dict'
                                                                             ' per table.')

    def _add_train_args(short: str, descr: str):
        train_group = group.add_argument_group(f'{descr} model training arguments')
        train_group.add_argument(f'--default_{short}_train_args', default=None, type=str,
                                 help='Default arguments saved as JSON file.')
        train_group.add_argument(f'--default_{short}_train_epochs', default=None, type=int)
        train_group.add_argument(f'--default_{short}_train_batch_size', default=None, type=int)
        train_group.add_argument(f'--default_{short}_train_shuffle', default=None, type=bool)
        train_group.add_argument(f'--default_{short}_train_save_freq', default=None, type=int)
        train_group.add_argument(f'--default_{short}_train_resume', default=None, type=bool)
        train_group.add_argument(f'--{short}_train_args', type=str, help='Arguments saved as JSON files as a dict'
                                                                         ' per table.')

    _add_trainer_args('tab', 'Tabular')
    _add_trainer_args('deg', 'Degree')
    _add_train_args('tab', 'Tabular')
    _add_train_args('deg', 'Degree')


def _parse_generate_args(parser: ArgumentParser):
    group = parser.add_argument_group('Generate')
    group.add_argument('--skip_generate', default=True, action='store_false', dest='do_generate',
                       help='Whether to skip generating process.')
    group.add_argument('--save_generated_to', default='generated', type=str,
                       help='Save generated tables to the directory.')
    group.add_argument('--default_scaling', default=1, type=int, help='Default scaling factor.')
    group.add_argument('--scaling', type=str, nargs='*', default=[],
                       help='Scaling factors, where each non-default scaling factor is represented by 2-tuple'
                            'of table name and the factor. For example, set scaling factor of "t1" to 0.3, give'
                            '"--scaling t1 0.3".')
    group.add_argument('--default_gen_tab_bs', default=32, type=int,
                       help='Default batch size for tabular models in generation.')
    group.add_argument('--gen_tab_bs', type=str, nargs='*', default=[], help='Tabular model batch sizes in '
                                                                             'similar format as scaling.')
    group.add_argument('--default_gen_deg_bs', default=32, type=int,
                       help='Default batch size for degree models in generation.')
    group.add_argument('--gen_deg_bs', type=str, nargs='*', default=[], help='Degree model batch sizes in '
                                                                             'similar format as scaling.')
    group.add_argument('--save_synth_db', default=None, type=str, help='Save synthetic database with internal '
                                                                       'information to directory.')


def _parse_eval_args(parser: ArgumentParser):
    db_group = parser.add_argument_group('database')
    db_group.add_argument('--real_db_dir', required=True, type=str,
                          help='Path of the directory where the real database is saved.')
    db_group.add_argument('--fake_db_dir', required=True, type=str, nargs='+',
                          help='Path of the directory where the synthetic database is saved.')
    db_group.add_argument('--fake_db_names', type=str, nargs='+',
                          help='Names/version of synthetic databases. If provided, it should match `--fake_db_dir` in '
                               'order. If not provided, the directory names are used as database names.')

    constructor_group = parser.add_argument_group('constructor arguments')
    constructor_group.add_argument('--evaluator_path', type=str, default=None,
                                   help='JSON file holding the SyntheticDatabaseEvaluator constructor argument values. '
                                        'ForeignKey is described as a `dict` of its constructor\'s arguments. '
                                        'Some values read from the file can be overridden by CLI arguments (override '
                                        'by giving the CLI arguments some input). Some arguments in CLI are suffixed '
                                        'by `_from_file`, which typically involve lengthy input and hence the content '
                                        'are read from JSON file instead of directly from CLI.')
    constructor_group.add_argument('--eval_tables', type=bool, default=None)
    constructor_group.add_argument('--eval_parent_child', type=bool, default=None)
    constructor_group.add_argument('--eval_joined', type=bool, default=None)
    constructor_group.add_argument('--eval_queries', type=bool, default=None)
    constructor_group.add_argument('--tables', type=str, nargs='*', default=[])
    constructor_group.add_argument('--parent_child_pairs_from_file', type=str, default=None)
    constructor_group.add_argument('--all_direct_parent_child', type=bool, default=None)
    constructor_group.add_argument('--queries_from_file', type=str, default=None)
    constructor_group.add_argument('--query_args_from_file', type=str, default=None)
    constructor_group.add_argument('--save_tables_to', type=str, default=None)
    constructor_group.add_argument('--tabular_args_from_file', type=str, default=None)
    constructor_group.add_argument('--default_args_from_file', type=str, default=None)

    evaluate_group = parser.add_argument_group('evaluate arguments')
    evaluate_group.add_argument('--evaluate_path', type=str, default=None,
                                help='JSON file holding the SyntheticDatabaseEvaluator.evaluate argument values. '
                                     'Some can be overridden in similar manners to constructor.')
    evaluate_group.add_argument('--mean', type=str, default=None)
    evaluate_group.add_argument('--smooth', type=float, default=None)
    evaluate_group.add_argument('--visualize_args_from_file', type=str, default=None)

    save_group = parser.add_argument_group('save results path arguments')
    save_group.add_argument('--save_eval_res_to', type=str, default=None)
    save_group.add_argument('--save_complete_result_to', type=str, default=None)
    save_group.add_argument('--save_synthetic_tables_to', type=str, default=None)
    save_group.add_argument('--save_visualization_to', type=str, default=None)
    save_group.add_argument('--save_all_res_to', type=str, default='evaluation',
                            help='Path of directory to save eventual result of the evaluation.')


def parse_args() -> Namespace:
    """
    Parse arguments.

    **Return**: Parser arguments.
    """
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='op')

    train_gen_parser = subparsers.add_parser('train_gen')
    _parse_distributed_args(train_gen_parser)
    _parse_database_args(train_gen_parser)
    _parse_train_args(train_gen_parser)
    _parse_generate_args(train_gen_parser)

    eval_parser = subparsers.add_parser('evaluate')
    _parse_eval_args(eval_parser)

    parser.add_argument('--seed', type=int, default=None, help='Fix seed before training for reproduction if provided.')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level.',
                        choices=['NOTSET', 'DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'FATAL', 'CRITICAL'])
    parser.add_argument('--temp_cache', type=str, default='.temp', help='Directory to hold temporary cache files.')
    parser.add_argument('--num_processes', type=int, default=200,
                        help='Number of processes to run in parallel maximally.')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training group.')

    return parser.parse_args()


def _cfgfile2argdict(default_path: Optional[str], sep_path: Optional[str], args: Namespace, prefix: str) -> DefaultDict:
    default_args = {}
    if default_path is not None and os.path.exists(default_path):
        with open(default_path, 'r') as f:
            default_args = json.load(f)
    sep_args = {}
    if sep_path is not None and os.path.exists(sep_path):
        with open(sep_path, 'r') as f:
            sep_args = json.load(f)

    prefix = f'default_{prefix}_'
    prefix_len = len(prefix)
    for n, v in args.__dict__.items():
        if n.startswith(prefix) and v is not None:
            default_args[n[prefix_len:]] = v

    return defaultdict(lambda: default_args, sep_args)


def _narg2nbdict(default_value: Any, special: List[str], vtype: type) -> DefaultDict:
    dict_from_list = {}
    assert len(special) % 2 == 0, 'Length of this argument should be even.'
    for n, v in zip(special[::2], special[1::2]):
        dict_from_list[n] = vtype(v)
    return defaultdict(lambda: default_value, dict_from_list)


def _fix_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def _train_gen(args: Namespace):
    init_process_group(args.distributed, args.num_processes)

    augmented_db = engine.augment(
        file_path=args.db_config_path, engine=args.engine,
        data_dir=args.data_dir, mtype=args.mtype,
        save_db_to=args.db_dir_path, resume=args.aug_resume,
        temp_cache=args.temp_cache
    )
    _LOGGER.info('Finished loading database.')

    tab_models, deg_models = engine.train(
        database=augmented_db, do_train=args.do_train,
        tab_trainer_args=_cfgfile2argdict(args.default_tab_trainer_args, args.tab_trainer_args, args, 'tab_trainer'),
        deg_trainer_args=_cfgfile2argdict(args.default_deg_trainer_args, args.deg_trainer_args, args, 'deg_trainer'),
        tab_train_args=_cfgfile2argdict(args.default_tab_train_args, args.tab_train_args, args, 'tab_train'),
        deg_train_args=_cfgfile2argdict(args.default_deg_train_args, args.deg_train_args, args, 'deg_train')
    )
    _LOGGER.info('Finished loading models.')

    if args.do_generate:
        synthetic_db = engine.generate(
            real_db=augmented_db, tab_models=tab_models, deg_models=deg_models,
            save_to=args.save_generated_to,
            scaling=_narg2nbdict(args.default_scaling, args.scaling, float),
            tab_batch_sizes=_narg2nbdict(args.default_gen_tab_bs, args.gen_tab_bs, int),
            deg_batch_sizes=_narg2nbdict(args.default_gen_deg_bs, args.gen_deg_bs, int),
            save_db_to=args.save_synth_db, temp_cache=args.temp_cache
        )


def _evaluate(args: Namespace):
    if args.fake_db_names is None:
        args.fake_db_names = [os.path.basename(name) for name in args.fake_db_dir]
    assert len(args.fake_db_names) == len(args.fake_db_dir), 'Number of directories for synthetic databases ' \
                                                             'should match the number of names provided.'
    synthetic_db = {
        name: dir_path for name, dir_path in zip(args.fake_db_names, args.fake_db_dir)
    }

    constructor_args = {}
    if args.evaluator_path is not None:
        with open(args.evaluator_path, 'r') as f:
            constructor_args = json.load(f)
    for n, v in args.__dict__.items():
        if v is None:
            continue
        if n not in {'eval_tables', 'eval_parent_child', 'eval_joined', 'eval_queries', 'tables',
                     'parent_child_pairs_from_file', 'all_direct_parent_child', 'queries_from_file',
                     'query_args_from_file', 'save_tables_to', 'tabular_args_from_file', 'default_args_from_file'}:
            continue
        if n.endswith('_from_file'):
            with open(v, 'r') as f:
                v = json.load(f)
        constructor_args[n] = v

    eval_args = {}
    if args.evaluate_path is not None:
        with open(args.evaluate_path, 'r') as f:
            eval_args = json.load(f)
    for n, v in args.__dict__.items():
        if v is None:
            continue
        if n not in {'mean', 'smooth', 'visualize_args_from_file'}:
            continue
        if n.endswith('_from_file'):
            with open(v, 'r') as f:
                v = json.load(f)
        eval_args[n] = v

    result = engine.evaluate(
        real=args.real_db_dir, synthetic=synthetic_db,
        constructor_args=constructor_args, eval_args=eval_args,
        save_eval_res_to=args.save_eval_res_to, save_complete_result_to=args.save_complete_result_to,
        save_synthetic_tables_to=args.save_synthetic_tables_to, save_visualization_to=args.save_visualization_to
    )
    os.makedirs(args.save_all_res_to, exist_ok=True)
    with open(os.path.join(args.save_all_res_to, 'result.pt'), 'wb') as f:
        pickle.dump(result, f)


def main():
    args = parse_args()
    if args.seed is not None:
        _fix_seed(args.seed)
    logging.basicConfig(level=args.log_level, format='%(asctime)s [%(levelname)s] - %(module)s : %(message)s')
    _LOGGER.debug(f'Command line arguments: {args}')

    if args.op == 'train_gen':
        _train_gen(args)
    elif args.op == 'evaluate':
        _evaluate(args)


if __name__ == '__main__':
    main()
