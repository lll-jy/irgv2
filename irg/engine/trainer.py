"""Train database generators."""

from collections import defaultdict
from typing import Optional, Dict, List, Tuple, DefaultDict
import logging

from torch import Tensor

from ..tabular import TabularTrainer
from ..schema import Database, Table
from ..tabular import create_trainer as create_tab_trainer

_LOGGER = logging.getLogger()


def train(database: Database, do_train: bool,
          tab_trainer_args: Optional[Dict[str, Dict]] = None, deg_trainer_args: Optional[Dict[str, Dict]] = None,
          tab_train_args: Optional[Dict[str, Dict]] = None, deg_train_args: Optional[Dict[str, Dict]] = None) -> \
        Tuple[Dict[str, TabularTrainer], Dict[str, TabularTrainer]]:
    """
    Train database generator.

    **Args**:

    - `database` (`Database`): Augmented database.
    - `do_train` (`bool`): Whether to do training.
    - `tab_trainer_args` (`Optional[Dict[str, Dict]]`): A dictionary describing every tabular generator's trainer
      constructor arguments where keys are table names and values are arguments to
      [trainer creator](../tabular#irg.tabular.create_trainer) except for dimension-related ones.
    - `deg_trainer_args` (`Optional[Dict[str, Dict]]`): Same as `tab_trainer_args` but for degree generation.
    - `tab_train_args` (`Optional[Dict[str, Dict]]`): A dictionary describing every tabular generator's trainer
      training arguments where keys are table names and values are arguments to
      [trainer](../utils#irg.utils.Trainer.train) except for data.
    - `deg_train_args` (`Optional[Dict[str, Dict]]`): Same as `tab_train_args` but for degree generation.

    **Return**: Tabular models as a dict, and degree models as a dict.
    """

    if not isinstance(tab_trainer_args, DefaultDict):
        tab_trainer_args = defaultdict(lambda: {}, tab_trainer_args)
    if not isinstance(deg_trainer_args, DefaultDict):
        deg_trainer_args = defaultdict(lambda: {}, deg_trainer_args)
    if not isinstance(tab_train_args, DefaultDict):
        tab_train_args = defaultdict(lambda: {}, tab_train_args)
    if not isinstance(deg_train_args, DefaultDict):
        deg_train_args = defaultdict(lambda: {}, deg_train_args)
    _LOGGER.debug('Finished constructing arguments.')

    tabular_models, deg_models = {}, {}
    for name, table in database.tables:
        table = Table.load(table)
        if table.ttype == 'base':
            continue
        tabular_known, tabular_unknown, cat_dims = table.ptg_data
        tabular_models[name] = _train_model(tabular_known, tabular_unknown, cat_dims, do_train,
                                            tab_trainer_args[name], tab_train_args[name], name)
        _LOGGER.debug(f'Loaded tabular model for {name}.')

        if not table.is_independent:
            deg_known, deg_unknown, cat_dims = table.deg_data
            deg_models[name] = _train_model(deg_known, deg_unknown, cat_dims, do_train,
                                            deg_trainer_args[name], deg_train_args[name], name)
            _LOGGER.debug(f'Loaded degree model for {name}.')

    return tabular_models, deg_models


def _train_model(known: Tensor, unknown: Tensor, cat_dims: List[Tuple[int, int]], do_train: bool,
                 trainer_args: Dict, train_args: Dict, descr: str) -> TabularTrainer:
    known_dim, unknown_dim = known.shape[1], unknown.shape[1]
    trainer = create_tab_trainer(cat_dims=cat_dims, known_dim=known_dim, unknown_dim=unknown_dim, descr=descr,
                                 **trainer_args)
    if do_train:
        trainer.train(known, unknown, **train_args)
    return trainer
