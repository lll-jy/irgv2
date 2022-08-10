"""Train database generators."""

from collections import defaultdict
from typing import Optional, OrderedDict, Dict, List, Tuple, DefaultDict

from torch import Tensor

from ..schema.database import create as create_db
from ..tabular import create_trainer as create_tab_trainer
from ..utils import Trainer


def train(schema: Optional[OrderedDict] = None, file_path: Optional[str] = None, engine: Optional[str] = None,
          data_dir: str = '.', mtype: str = 'unrelated',
          tab_trainer_args: Optional[Dict[str, Dict]] = None, deg_trainer_args: Optional[Dict[str, Dict]] = None,
          tab_train_args: Optional[Dict[str, Dict]] = None, deg_train_args: Optional[Dict[str, Dict]] = None):
    """
    Train database generator.

    **Args**:

    - `schema` to `mtype`: Arguments to [database creator](../schema/database#irg.schema.database.create)
    - `tab_trainer_args` (`Optional[Dict[str, Dict]]`): A dictionary describing every tabular generator's trainer
      constructor arguments where keys are table names and values are arguments to
      [trainer creator](../tabular#irg.tabular.create_trainer) except for dimension-related ones.
    - `deg_trainer_args` (`Optional[Dict[str, Dict]]`): Same as `tab_trainer_args` but for degree generation.
    - `tab_train_args` (`Optional[Dict[str, Dict]]`): A dictionary describing every tabular generator's trainer
      training arguments where keys are table names and values are arguments to
      [trainer](../utils#irg.utils.Trainer.train) except for data.
    - `deg_train_args` (`Optional[Dict[str, Dict]]`): Same as `tab_train_args` but for degree generation.
    """
    database = create_db(schema, file_path, engine, data_dir, mtype)
    database.augment()

    if not isinstance(tab_trainer_args, DefaultDict):
        tab_trainer_args = defaultdict(lambda: {}, tab_trainer_args)
    deg_trainer_args = defaultdict(lambda: {}, deg_trainer_args)
    tab_train_args = defaultdict(lambda: {}, tab_train_args)
    deg_train_args = defaultdict(lambda: {}, deg_train_args)

    tabular_models, deg_models = {}, {}
    for name, table in database.tables:
        tabular_known, tabular_unknown, cat_dims = table.ptg_data
        tabular_models[name] = _train_model(tabular_known, tabular_unknown, cat_dims,
                                            tab_trainer_args[name], tab_train_args[name])

        if not table.is_independent:
            deg_known, deg_unknown, cat_dims = table.deg_data
            deg_models[name] = _train_model(deg_known, deg_unknown, cat_dims,
                                            deg_trainer_args[name], deg_train_args[name])


def _train_model(known: Tensor, unknown: Tensor, cat_dims: List[Tuple[int, int]],
                 trainer_args: Dict, train_args: Dict) -> Trainer:
    known_dim, unknown_dim = known.shape[1], unknown.shape[1]
    trainer = create_tab_trainer(cat_dims=cat_dims, known_dim=known_dim, unknown_dim=unknown_dim, **trainer_args)
    trainer.train(known, unknown, **train_args)
    return trainer
