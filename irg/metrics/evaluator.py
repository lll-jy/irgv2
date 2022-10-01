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
from typing import Optional, List, Dict, Any, DefaultDict, Union
import os
import pickle

import pandas as pd

from ..schema import Database, SyntheticDatabase, Table
from ..schema.database.base import ForeignKey
from .tabular import SyntheticTableEvaluator, TableVisualizer

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
                 save_tables_to: str = 'eval_tables',
                 tabular_args: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
                 default_args: Optional[Dict[str, Any]] = None):
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
        - `tabular_args` (`Optional[Dict[str, Dict[str, Dict[str, Any]]]]`): Arguments to
          [`SyntheticTableEvaluator`](./tabular#irg.metrics.tabular.SyntheticTableEvaluator)
          for each set of tabular data. The first level keys are the names of the four tabular types.
          The second level keys are the names of the tables following the naming mentioned above.
          The values in the second level are arguments to `SyntheticTableEvaluator` except for `save_to`.
        - `default_args` (`Optional[Dict[str, Any]]`): Default arguments to
          [`SyntheticTableEvaluator`](./tabular#irg.metrics.tabular.SyntheticTableEvaluator) (second level values
          for `tabular_args` if the relevant keys are not found.
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

        self._table_dir = save_tables_to
        os.makedirs(self._table_dir, exist_ok=True)
        self._real_tables = self._construct_tables(real, 'real')

        eval_args = defaultdict(lambda: defaultdict(lambda: default_args))
        for type_descr, tables_in_type in tabular_args.items():
            for table_descr, table_args in tables_in_type.items():
                eval_args[type_descr][table_descr] |= table_args
        self._evaluators = {}
        os.makedirs(os.path.join(self._table_dir, 'complete'), exist_ok=True)
        os.makedirs(os.path.join(self._table_dir, 'complete', 'real'), exist_ok=True)
        for type_descr, tables_in_type in self._real_tables.items():
            evaluators = {}
            os.makedirs(os.path.join(self._table_dir, 'complete', 'real', type_descr), exist_ok=True)
            for table_descr, table in tables_in_type.items():
                evaluator = SyntheticTableEvaluator(**eval_args[type_descr][table_descr])
                evaluators[table_descr] = evaluator
                table = Table.load(table)
                table.save_complete(os.path.join(self._table_dir, 'complete', 'real', type_descr, table_descr))
            self._evaluators[type_descr] = evaluators

    def _construct_tables(self, db: Database, db_descr: str) -> Dict[str, Dict[str, str]]:
        result = {}
        os.makedirs(os.path.join(self._table_dir, 'cache'), exist_ok=True)
        os.makedirs(os.path.join(self._table_dir, 'cache', db_descr), exist_ok=True)

        if self._eval_tables:
            tables = self._tables if self._tables is not None else [name for name, _ in db.tables]
            os.makedirs(os.path.join(self._table_dir, 'cache', db_descr, 'tables'), exist_ok=True)
            result['tables'] = {}
            for table in tables:
                saved_path = os.path.join(self._table_dir, 'cache', db_descr, 'tables', f'{table}.pkl')
                db[table].save(saved_path)
                result['tables'][table] = saved_path
            _LOGGER.debug(f'Constructed tables for {db_descr}.')

        if self._eval_parent_child:
            os.makedirs(os.path.join(self._table_dir, 'cache', db_descr, 'parentchild'), exist_ok=True)
            result['parent child'] = {}
            for i, fk in enumerate(self._all_fk):
                descr = f'{i}__{fk.child}__{fk.parent}'
                saved_path = os.path.join(self._table_dir, 'cache', db_descr, 'parentchild', f'{descr}.pkl')
                db.join(fk).save(saved_path)
                result['parent child'][descr] = saved_path
            _LOGGER.debug(f'Construct parent-child joined tables for {db_descr}.')

        if self._eval_joined:
            os.makedirs(os.path.join(self._table_dir, 'cache', db_descr, 'joined'), exist_ok=True)
            saved_path = os.path.join(self._table_dir, 'cache', db_descr, 'joined', 'joined.pkl')
            db.all_joined.save(saved_path)
            result['joined'] = {'joined': saved_path}
            _LOGGER.debug(f'Construct all-joined table for {db_descr}.')

        if self._eval_queries:
            os.makedirs(os.path.join(self._table_dir, 'cache', db_descr, 'queries'), exist_ok=True)
            result['queries'] = {}
            for descr, query in self._queries:
                saved_path = os.path.join(self._table_dir, 'cache', db_descr, 'queries', f'{descr}.pkl')
                db.query(query, descr, **self._query_args[descr]).save(saved_path)
                result['queries'][descr] = saved_path
            _LOGGER.debug(f'Construct query tables for {db_descr}.')

        _LOGGER.info(f'Constructed all tables for evaluation for {db_descr}.')

        return result

    def evaluate(self, synthetic: SyntheticDatabase, descr: str, mean: str = 'arithmetic', smooth: float = 0.1,
                 save_eval_res_to: Optional[str] = None, save_complete_result_to: Optional[str] = None,
                 save_visualization_to: Optional[str] = None,
                 visualize_args: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
        """
        Evaluate synthetic database.

        **Args**:

        - `synthetic` (`SyntheticDatabase`): Synthetic database.
        - `descr` (`str`): Description of the synthetic database.
        - `mean` and `smooth`: Arguments to
          [`SyntheticTableEvaluator.summary`](./tabular#irg.metrics.tabular.SyntheticTableEvaluator.summary).
        - `save_eval_res_to` (`Optional[str]`): Path to save extra evaluation result that is not returned.
          Not saved if not provided.
        - `save_complete_result_to` (`Optional[str]`): Path to save complete evaluation results for each tabular data
          set to. If it is not provided, this piece of information is not saved.
        - `save_visualization_to` (`Optional[str]`): If provided, all constructed tabular data sets are visualized and
          saved to the designated directory.
        - `visualize_args` (`Optional[Dict[str, Dict[str, Any]]]`): Visualization arguments. For each pair of real and
          synthetic tabular data, the same set of visualization arguments are applied, but multiple sets can be applied
          together. If not provided, a default setting is still run. The keys of this argument serves as `descr`, so
          this argument does not need to be specified in its values, otherwise there will be errors. Also, `save_dir`
          should not be specified because they are saved in designated place under `save_visualization_to`.

        **Return**: A `pd.DataFrame` describing the metrics result.
        """
        synthetic_tables = self._construct_tables(synthetic, descr)
        if save_eval_res_to is not None:
            os.makedirs(save_eval_res_to, exist_ok=True)
        if save_visualization_to is not None:
            os.makedirs(save_visualization_to, exist_ok=True)
        visualize_args = visualize_args if visualize_args is not None else {'default': {}}

        results, summary = {}, {}
        os.makedirs(os.path.join(self._table_dir, 'complete', descr), exist_ok=True)
        for type_descr, evaluators_in_type in self._evaluators.items():
            print('evaluate type', type_descr)
            type_results, type_summary = {}, {}
            if save_eval_res_to is not None:
                os.makedirs(os.path.join(save_eval_res_to, type_descr), exist_ok=True)
            if save_visualization_to is not None:
                os.makedirs(os.path.join(save_visualization_to, type_descr), exist_ok=True)
            os.makedirs(os.path.join(self._table_dir, 'complete', descr, type_descr), exist_ok=True)

            for table_descr, evaluator in evaluators_in_type.items():
                print('evaluate table', table_descr)
                real_table = self._real_tables[type_descr][table_descr]
                real_table = Table.load(real_table)
                synthetic_table = synthetic_tables[type_descr][table_descr]
                synthetic_table = Table.load(synthetic_table)
                synthetic_table.save_complete(os.path.join(self._table_dir, 'complete', descr, type_descr, table_descr))
                evaluator.evaluate(real_table, synthetic_table,
                                   os.path.join(save_eval_res_to, type_descr, table_descr)
                                   if save_eval_res_to is not None else None)
                type_results[table_descr] = evaluator.result
                type_summary[table_descr] = evaluator.summary(mean, smooth)

                _LOGGER.info(f'Finished evaluating {type_descr} table {table_descr}.')

                if save_visualization_to is not None:
                    visualizer = TableVisualizer(real_table, synthetic_table)
                    vis_dir = os.path.join(save_visualization_to, type_descr, table_descr)
                    os.makedirs(vis_dir, exist_ok=True)
                    for descr, args in visualize_args.items():
                        visualizer.visualize(descr=descr, save_dir=vis_dir, **args)
                        _LOGGER.debug(f'Finished visualizing {type_descr} table {table_descr} version {descr}.')
                    _LOGGER.info(f'Finished visualizing {type_descr} table {table_descr}.')
            results[type_descr] = type_results
            summary[type_descr] = pd.DataFrame(type_summary)
            _LOGGER.info(f'Finished evaluating {type_descr}.')

        if save_complete_result_to is not None:
            with open(save_complete_result_to, 'wb') as f:
                pickle.dump(results, f)
        result = pd.concat(summary, axis=1)
        _LOGGER.info(f'Finished evaluating database {descr}.')
        return result
