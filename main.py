"""Main runner script."""

from argparse import ArgumentParser, Namespace
from collections import defaultdict
import json
import os
from typing import Optional, DefaultDict, Any, List

from irg.utils.dist import init_process_group
from irg import engine


def _parse_distributed_args(parser: ArgumentParser):
    group = parser.add_argument_group('Distributed')
    group.add_argument('--distributed', default=False, action='store_true',
                       help='Whether this is distributed training.')
    group.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training group.')


def _parse_database_args(parser: ArgumentParser):
    group = parser.add_argument_group('Database')
    group.add_argument('--db_config_path', default='config.json', type=str,
                       help='File path that content of this database schema is saved.')
    group.add_argument('--engine', default='json', type=str, help='File format of database',
                       choices=['json', 'pickle', 'yaml', 'torch'])
    group.add_argument('--data_dir', default='.', type=str, help='Data directory with data content.')
    group.add_argument('--mtype', default='unrelated', type=str, help='Augmentation mechanism of database.',
                       choices=['unrelated'])  # TODO: more types
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


def parse_args() -> Namespace:
    """
    Parse arguments.

    **Return**: Parser arguments.
    """
    parser = ArgumentParser()
    _parse_distributed_args(parser)
    _parse_database_args(parser)
    _parse_train_args(parser)
    _parse_generate_args(parser)
    return parser.parse_args()


def _cfgfile2argdict(default_path: Optional[str], sep_path: str, args: Namespace, prefix: str) -> DefaultDict:
    default_args = {}
    if default_path is not None and os.path.exists(default_path):
        with open(default_path, 'r') as f:
            default_args = json.load(f)

    prefix = f'default_{prefix}_'
    prefix_len = len(prefix)
    for n, v in args.__dict__.items():
        if n.startswith(prefix) and v is not None:
            default_args[n[prefix_len:]] = v

    return defaultdict(lambda: default_args, sep_path)


def _narg2nbdict(default_value: Any, special: List[str], vtype: type) -> DefaultDict:
    dict_from_list = {}
    assert len(special) % 2 == 0, 'Length of this argument should be even.'
    for n, v in zip(special[::2], special[1::2]):
        dict_from_list[n] = vtype(v)
    return defaultdict(lambda: default_value, dict_from_list)


def main():
    args = parse_args()
    if args.distributed:
        init_process_group()

    augmented_db = engine.augment(
        file_path=args.db_config_path, engine=args.engine,
        data_dir=args.data_dir, mtype=args.mtype,
        save_db_to=args.db_dir_path, resume=args.aug_resume
    )

    tab_models, deg_models = engine.train(
        database=augmented_db, do_train=args.do_train,
        tab_trainer_args=_cfgfile2argdict(args.default_tab_trainer_args, args.tab_trainer_args, args, 'tab_trainer'),
        deg_trainer_args=_cfgfile2argdict(args.default_deg_trainer_args, args.deg_trainer_args, args, 'deg_trainer'),
        tab_train_args=_cfgfile2argdict(args.default_tab_train_args, args.tab_train_args, args, 'tab_train'),
        deg_train_args=_cfgfile2argdict(args.default_deg_train_args, args.deg_train_args, args, 'deg_train')
    )

    if args.do_generate:
        synthetic_db = engine.generate(
            real_db=augmented_db, tab_models=tab_models, deg_models=deg_models,
            save_to=args.save_generated_to,
            scaling=_narg2nbdict(args.default_scaling, args.scaling, float),
            tab_batch_sizes=_narg2nbdict(args.default_gen_tab_bs, args.gen_tab_bs, int),
            deg_batch_sizes=_narg2nbdict(args.default_gen_deg_bs, args.gen_deg_bs, int),
            save_db_to=args.save_synth_db
        )


if __name__ == '__main__':
    main()
