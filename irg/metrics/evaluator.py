"""
Evaluate based on tabular data extracted from database.
There are four types of tabular data that can be extracted:

- `tables`: Each table in the database. Each set of tabular data is named after the table's own name.
- `parent child`: Two tables joined by some (direct or indirect) foreign keys. Each set of tabular data is named by
  `{FK_ID}__{CHILD_NAME}__{PARENT_NAME}`, where foreign key IDs are cumulative index of foreign keys in the database.
  For example, a database with tables T1, T2, T3 (ordered) with 0, 2, 1 foreign keys respectively, the index of the
  foreign key from T3 is 2 (0-based). If another set of size 3 if provided manually, then the indices are shifted
  accordingly. Namely, index 2 becomes 5.
- `joined`: All tables in the database joined by using all foreign keys. There is only one table in this group, and its
  name is also `joined`.
- `queries`: Tables constructed by some SQL queries (arbitrary query applicable in this database). Names of tabular data
  in this set are up to the user to specify.
"""
import logging
from collections import defaultdict
from typing import Optional, List, Dict, Any, DefaultDict, Union, Literal
import os
import pickle

import pandas as pd

from ..schema import Database, SyntheticDatabase, Table, SyntheticTable
from ..schema.database.base import ForeignKey
from .tabular import SyntheticTableEvaluator, TableVisualizer
from .tabular.visualize import create_visualizer
from ..utils.misc import calculate_mean

_LOGGER = logging.getLogger()


class SyntheticDatabaseEvaluator:
    """Evaluator for synthetic database generation on tabular data extracted from the database."""
    def __init__(self, real: Database,
                 eval_tables: bool = True, eval_parent_child: bool = True, eval_joined: bool = False,
                 eval_queries: bool = True,
                 tables: Optional[List[str]] = None,
                 parent_child_pairs: Optional[List[Union[ForeignKey, Dict[str, Any]]]] = None,
                 all_direct_parent_child: bool = True,
                 queries: Optional[Dict[str, str]] = None, query_args: Optional[Dict[str, Dict[str, Any]]] = None,
                 save_tables_to: str = 'eval_tables', save_eval_res_to: str = 'eval_res', save_vis_to: str = 'vis',
                 tabular_args: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
                 default_args: Optional[Dict[str, Any]] = None, visualize_args: Optional[Dict[str, Any]] = None):
        """
        **Args**:

        - `real` (`Database`): The real database.
        - `eval_tables`, `eval_parent_child`, `eval_joined`, `eval_queries` (`bool`): Whether to apply tabular
          evaluation on this set of tabular data extracted from the database.
          Default is T, T, F, F.
        - `tables` (`Optional[List[str]]`): The set of table names in `tables` tabular data type. If not provided,
          all tables are evaluated.
        - `parent_child_pairs` (`Optional[List[Union[ForeignKey, Dict[str, Any]]]]`): List of foreign keys to join to
          construct parent-child pairs. If `Dict` is provided instead of `ForeignKey`, construct foreign key by using
          the content of `Dict` as constructor arguments.
        - `all_direct_parent_child` (`bool`): Whether include all existing direct foreign keys in the database
          in `parent child` type of tabular data. This will be executed in addition to `parent_child_pairs`.
          Default is `True`.
        - `queries` (`Optional[Dict[str, str]]`): Queries, each with a short description as key to the `dict`, to
          construct tables in the `queries` type.
        - `query_args` (`Optional[Dict[str, Dict[str, Any]]]`): `kwargs` to
          [`Database.query`](../schema/database/base#irg.schema.database.base.Database.query) per query if needed.
          Keys of the `dict` should match the keys of `queries` if provided.
        - `save_tables_to` (`str`): Path to save the constructed tabular data in the format of `Table` based
          on the real database. Default is `'eval_tables'`.
        - `save_eval_res_to` (`str`): Path to save the evaluation results to. Default is `'eval_res'`.
        - `save_vis_to` (`str`): Path to save visualization result to. Default is `'vis'`.
        - `tabular_args` (`Optional[Dict[str, Dict[str, Dict[str, Any]]]]`): Arguments to
          [`SyntheticTableEvaluator`](./tabular#irg.metrics.tabular.SyntheticTableEvaluator)
          for each set of tabular data. The first level keys are the names of the four tabular types.
          The second level keys are the names of the tables following the naming mentioned above.
          The values in the second level are arguments to `SyntheticTableEvaluator` except for `save_to`.
        - `default_args` (`Optional[Dict[str, Any]]`): Default arguments to
          [`SyntheticTableEvaluator`](./tabular#irg.metrics.tabular.SyntheticTableEvaluator) (second level values
          for `tabular_args` if the relevant keys are not found.
        - `visualize_args` (`Optional[Dict[str, Dict[str, Any]]`): Visualization settings applying to all tables, where
          keys are short descriptions of the visualization and values are arguments to
          [`create_visualizer`](./tabular#irg.metrics.tabular.visualize.create_visualizer) constructor.
          By default, we will use PCA and LDA.
        """
        self._real = real

        (self._eval_tables, self._eval_parent_child,
         self._eval_joined, self._eval_queries) = eval_tables, eval_parent_child, eval_joined, eval_queries
        self._tables = tables
        self._all_fk = ([] if parent_child_pairs is None else [
            fk if isinstance(fk, ForeignKey) else ForeignKey(**fk) for fk in parent_child_pairs
        ]) + (real.foreign_keys if all_direct_parent_child else [])
        self._queries = queries if queries is not None else {}
        self._query_args = query_args if isinstance(query_args, DefaultDict) else defaultdict(dict, query_args) \
            if query_args is not None else defaultdict(dict)
        tabular_args = tabular_args if tabular_args is not None else {}
        default_args = default_args if default_args is not None else {}
        self._vis_args = visualize_args if visualize_args is not None else {
            'pca': {'policy': 'pca'},
            # 'lda': {'policy': 'lda'}
        }

        self._table_dir = save_tables_to
        os.makedirs(self._table_dir, exist_ok=True)
        self._real_tables = self._construct_tables(real, 'real')

        eval_args = defaultdict(lambda: defaultdict(lambda: default_args.copy()))
        for type_descr, tables_in_type in tabular_args.items():
            for table_descr, table_args in tables_in_type.items():
                eval_args[type_descr][table_descr] |= table_args
        self._evaluators, self._visualizers = {}, {}

        self._res_dir = save_eval_res_to
        os.makedirs(self._res_dir, exist_ok=True)
        self._vis_dir = save_vis_to
        os.makedirs(self._vis_dir, exist_ok=True)

        for type_descr, tables_in_type in self._real_tables.items():
            evaluators, visualizers = {}, {}
            os.makedirs(os.path.join(self._res_dir, type_descr), exist_ok=True)
            os.makedirs(os.path.join(self._vis_dir, type_descr), exist_ok=True)
            for table_descr, table in tables_in_type.items():
                table = Table.load(table)
                evaluator = SyntheticTableEvaluator(
                    real=table, res_dir=os.path.join(self._res_dir, type_descr, table_descr),
                    **eval_args[type_descr][table_descr]
                )
                evaluators[table_descr] = evaluator
                os.makedirs(os.path.join(self._vis_dir, type_descr, table_descr), exist_ok=True)
                visualizers[table_descr] = {
                    descr: create_visualizer(
                        real=table, model_dir=os.path.join(self._vis_dir, type_descr, table_descr, 'models'),
                        vis_to=os.path.join(self._vis_dir, type_descr, table_descr, 'vis'), **vis_args
                    ) for descr, vis_args in self._vis_args.items()
                }
            self._evaluators[type_descr] = evaluators
            self._visualizers[type_descr] = visualizers

    def _construct_tables(self, db: Database, db_descr: str) -> Dict[str, Dict[str, str]]:
        result = {}
        os.makedirs(os.path.join(self._table_dir, db_descr), exist_ok=True)

        if self._eval_tables:
            tables = self._tables if self._tables is not None else [name for name, _ in db.tables()]
            os.makedirs(os.path.join(self._table_dir, db_descr, 'tables'), exist_ok=True)
            result['tables'] = {}
            for table in tables:
                saved_path = os.path.join(self._table_dir, db_descr, 'tables', f'{table}.pkl')
                db[table].save(saved_path)
                result['tables'][table] = saved_path
            _LOGGER.debug(f'Constructed tables for {db_descr}.')

        if self._eval_parent_child:
            os.makedirs(os.path.join(self._table_dir, db_descr, 'parentchild'), exist_ok=True)
            result['parent child'] = {}
            for i, fk in enumerate(self._all_fk):
                descr = f'{i}__{fk.child}__{fk.parent}'
                saved_path = os.path.join(self._table_dir, db_descr, 'parentchild', f'{descr}.pkl')
                db.join(fk).save(saved_path)
                result['parent child'][descr] = saved_path
            _LOGGER.debug(f'Construct parent-child joined tables for {db_descr}.')

        if self._eval_joined:
            os.makedirs(os.path.join(self._table_dir, db_descr, 'joined'), exist_ok=True)
            saved_path = os.path.join(self._table_dir, db_descr, 'joined', 'joined.pkl')
            db.all_joined.save(saved_path)
            result['joined'] = {'joined': saved_path}
            _LOGGER.debug(f'Construct all-joined table for {db_descr}.')

        if self._eval_queries:
            os.makedirs(os.path.join(self._table_dir, db_descr, 'queries'), exist_ok=True)
            result['queries'] = {}
            for descr, query in self._queries.items():
                saved_path = os.path.join(self._table_dir, db_descr, 'queries', f'{descr}.pkl')
                db.query(query, descr, **self._query_args[descr]).save(saved_path)
                result['queries'][descr] = saved_path
            _LOGGER.debug(f'Construct query tables for {db_descr}.')

        _LOGGER.info(f'Constructed all tables for evaluation for {db_descr}.')

        return result

    def evaluate(self, synthetic: SyntheticDatabase, descr: str, mean: str = 'arithmetic', smooth: float = 0.1,
                 save_complete_result_to: Optional[str] = None) -> pd.DataFrame:
        """
        Evaluate synthetic database.

        **Args**:

        - `synthetic` (`SyntheticDatabase`): Synthetic database.
        - `descr` (`str`): Description of the synthetic database.
        - `mean` and `smooth`: Arguments to
          [`SyntheticTableEvaluator.summary`](./tabular#irg.metrics.tabular.SyntheticTableEvaluator.summary).
        - `save_complete_result_to` (`Optional[str]`): Path to save complete evaluation results for each tabular data
          set to. If it is not provided, this piece of information is not saved.

        **Return**: A `pd.DataFrame` describing the metrics result.
        """
        synthetic_tables = self._construct_tables(synthetic, descr)

        results, summary = {}, {}
        for type_descr, evaluators_in_type in self._evaluators.items():
            type_results, type_summary = {}, {}

            for table_descr, evaluator in evaluators_in_type.items():
                synthetic_table = synthetic_tables[type_descr][table_descr]
                evaluator = self._evaluate_table(synthetic_table, descr, type_descr, table_descr, evaluator,
                                                 self._visualizers[type_descr][table_descr])
                type_results[table_descr] = evaluator.result()
                type_summary[table_descr] = evaluator.summary(mean, smooth)

            results[type_descr] = type_results
            summary[type_descr] = pd.DataFrame(type_summary)
            _LOGGER.info(f'Finished evaluating {type_descr}.')

        if save_complete_result_to is not None:
            os.makedirs(os.path.dirname(save_complete_result_to), exist_ok=True)
            with open(save_complete_result_to, 'wb') as f:
                pickle.dump(results, f)
        result = pd.concat(summary, axis=1)
        _LOGGER.info(f'Finished evaluating database {descr}.')
        return result

    def compare(self, info_level: Literal['all', 'type', 'table'] = 'all') -> pd.DataFrame:
        """
        Compare all evaluated tables in this evaluator.

        **Args**:

        - `info_level` (`Literal['all', 'type', 'table']`): Most detailed level of information to be returned. If too
          detailed information is available, we calculate a mean over them.

        **Return**: Full result of all evaluated tables consolidated in one dataframe. The column indices are table
        version descriptions, and row indices are multilevel with metric type, metric name, table type (if 'type' or
        'table' `info_level`), table name (if 'table' `info_type`).
        """
        final_res = {}
        for type_descr, evaluators_in_type in self._evaluators.items():
            for table_descr, evaluator in evaluators_in_type.items():
                compare_res = evaluator.compare(return_as='dict')
                for metric_type, metric_res in compare_res.items():
                    if metric_type not in final_res:
                        final_res[metric_type] = {}
                    for metric_name in metric_res.columns:
                        if metric_name not in final_res[metric_type]:
                            final_res[metric_type][metric_name] = {}
                        if type_descr not in final_res[metric_type][metric_name]:
                            final_res[metric_type][metric_name][type_descr] = {}
                        final_res[metric_type][metric_name][type_descr][table_descr] = metric_res[metric_name]

        all_combined = {}
        for metric_type, type_res in final_res.items():
            per_metric_type = {}
            for metric_name, metric_res in type_res.items():
                per_metric_sep = {}
                for type_descr, table_type_res in metric_res.items():
                    per_table_type = pd.DataFrame(table_type_res)
                    if info_level != 'table':
                        per_table_type = per_table_type.aggregate(lambda x: calculate_mean(x, 'harmonic', 0), axis=1)
                    per_metric_sep[type_descr] = per_table_type
                if info_level == 'table':
                    per_metric = pd.concat(per_metric_sep, axis=1)
                else:
                    per_metric = pd.DataFrame(per_metric_sep)
                if info_level == 'all':
                    per_metric = per_metric.aggregate(lambda x: calculate_mean(x, 'harmonic', 0), axis=1)
                per_metric_type[metric_name] = per_metric
            if info_level == 'all':
                per_metric_type = pd.DataFrame(per_metric_type)
            else:
                per_metric_type = pd.concat(per_metric_type, axis=1)
            all_combined[metric_type] = per_metric_type
        return pd.concat(all_combined, axis=1).T

    @staticmethod
    def _evaluate_table(synthetic_table: str, descr: str, type_descr: str, table_descr: str,
                        evaluator: SyntheticTableEvaluator, visualizers: Dict[str, TableVisualizer]) -> \
            SyntheticTableEvaluator:
        synthetic_table = SyntheticTable.load(synthetic_table)
        evaluator.evaluate(synthetic_table, descr)

        _LOGGER.info(f'Finished evaluating {type_descr} table {table_descr}.')

        for descr, visualizer in visualizers.items():
            visualizer.visualize(synthetic_table, descr)
            _LOGGER.debug(f'Finished visualizing {type_descr} table {table_descr} version {descr}.')
        _LOGGER.info(f'Finished visualizing {type_descr} table {table_descr}.')
        return evaluator
