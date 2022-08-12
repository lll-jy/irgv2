"""Evaluate a pair of real and fake databases."""

from typing import Union, Dict, Any, Optional
import os

import pandas as pd

from ..schema import Database, SyntheticDatabase
from ..metrics import SyntheticDatabaseEvaluator


def evaluate(real: Union[str, Database],
             synthetic: Union[Union[str, SyntheticDatabase], Dict[str, Union[str, SyntheticDatabase]]],
             constructor_args: Dict[str, Any], eval_args: Dict[str, Any],
             save_eval_res_to: Optional[str] = None, save_complete_result_to: Optional[str] = None,
             save_synthetic_tables_to: Optional[str] = None,
             save_visualization_to: Optional[str] = None,) -> Dict[str, pd.DataFrame]:
    """
    Evaluate a pair of real and fake databases.

    **Args**:

    - `real` (`Union[str, Database]`): Real database. If `str` is given, load from the directory
      as saved by [`Database.save_to_dir`](../schema#irg.schema.Database.save_to_dir).
    - `synthetic` (`Union[Union[str, SyntheticDatabase], Dict[str, Union[str, SyntheticDatabase]]]`):
      Synthetic database(s). If `str` is given, load from the directory
      as saved by [`Database.save_to_dir`](../schema#irg.schema.Database.save_to_dir).
      If `dict` is given, there are a number of synthetic databases to be evaluated, where keys are
      short descriptions.
    - `constructor_args` (`Dict[str, Any]`): Arguments to
      [`SyntheticDatabaseEvaluator`](../metrics#irg.metrics.SyntheticDatabaseEvaluator) constructor.
    - `eval_args` (`Dict[str, Any]`): Arguments to
      [`SyntheticDatabaseEvaluator.evaluate`](../metrics#irg.metrics.SyntheticDatabaseEvaluator.evaluate).
      Only `mean`, `smooth`, and `visualize_args` can be provided here.
    - `save_eval_res_to` (`Optional[str]`): Path to save extra evaluation result that is not returned.
          Not saved if not provided.
    - `save_complete_result_to` (`Optional[str]`): Path to save complete evaluation results for each tabular data
      set to. If it is not provided, this piece of information is not saved.
    - `save_synthetic_tables_to` (`Optional[str]`): Path to save constructed tabular data set of the synthetic
      database.
    - `save_visualization_to` (`Optional[str]`): If provided, all constructed tabular data sets are visualized and
      saved to the designated directory.

    **Return**: A dictionary of evaluation results where keys are description of synthetic databases and values
    are corresponding evaluation results.
    """
    if isinstance(real, str):
        real = Database.load_from_dir(real)
    if not isinstance(synthetic, Dict):
        synthetic = {'synthetic': synthetic}
    for descr, syn_db in synthetic.items():
        if isinstance(syn_db, str):
            synthetic[descr] = SyntheticDatabase.load_from_dir(syn_db)

    evaluator = SyntheticDatabaseEvaluator(real, **constructor_args)
    results = {
        descr: evaluator.evaluate(
            synthetic=syn_db,
            save_eval_res_to=None if save_eval_res_to is None else os.path.join(save_eval_res_to, descr),
            save_complete_result_to=None if save_complete_result_to is None else
            os.path.join(save_complete_result_to, descr),
            save_synthetic_tables_to=None if save_synthetic_tables_to is None else
            os.path.join(save_synthetic_tables_to, descr),
            save_visualization_to=None if save_visualization_to is None else os.path.join(save_visualization_to, descr),
            **eval_args
        )
        for descr, syn_db in synthetic.items()
    }
    return results
