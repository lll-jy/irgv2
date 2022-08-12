"""
Evaluate based on tabular data extracted from database.
There are four types of tabular data that can be extracted:

- `tables`: Each table in the database. Each set of tabular data is named after the table's own name.
- `parent child`: Two tables joined by some (direct or indirect) foreign keys. Each set of tabular data is named by
  `{FK_ID} : {CHILD_NAME} : {PARENT_NAME}`, where foreign key IDs are cumulative index of foreign keys in the database.
  For example, a database with tables T1, T2, T3 (ordered) with 0, 2, 1 foreign keys respectively, the index of the
  foreign key from T3 is 2 (0-based). If another set of size 3 if provided manually, then the indices are shifted
  accordingly. Namely, index 2 becomes 5.
- `joined`: All tables in the database joined by using all foreign keys. There is only one table in this group, and its
  name is empty string.
- `queries`: Tables constructed by some SQL queries (arbitrary query applicable in this database). Names of tabular data
  in this set are up to the user to specify.
"""
from collections import defaultdict
from typing import Optional, List, Dict, Any, DefaultDict
import os
import pickle

import pandas as pd

from ..schema import Database, SyntheticDatabase, Table
from ..schema.database.base import ForeignKey
from .tabular import SyntheticTableEvaluator


class SyntheticDatabaseEvaluator:
    """Evaluator for synthetic database generation on tabular data extracted from the database."""
    def __init__(self, real: Database,
                 eval_tables: bool = True, eval_parent_child: bool = True, eval_joined: bool = True,
                 eval_queries: bool = True,
                 tables: Optional[List[str]] = None,
                 parent_child_pairs: Optional[List[ForeignKey]] = None, all_direct_parent_child: bool = True,
                 queries: Optional[Dict[str, str]] = None, query_args: Optional[Dict[str, Dict[str, Any]]] = None,
                 save_eval_res_to: Optional[str] = None, save_tables_to: Optional[str] = None,
                 tabular_args: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
                 default_args: Optional[Dict[str, Any]] = None):
        """
        **Args**:

        - `real` (`Database`): The real database.
        - `eval_tables`, `eval_parent_child`, `eval_joined`, `eval_queries` (`bool`): Whether to apply tabular
          evaluation on this set of tabular data extracted from the database.
        - `tables` (`Optional[List[str]]`): The set of table names in `tables` tabular data type. If not provided,
          all tables are evaluated.
        - `parent_child_pairs` (`Optional[List[ForeignKey]]`): List of foreign keys to join to construct parent-child
          pairs.
        - `all_direct_parent_child` (`bool`): Whether include all existing direct foreign keys in the database
          in `parent child` type of tabular data. This will be executed in addition to `parent_child_pairs`.
          Default is `True`.
        - `queries` (`Optional[Dict[str, str]]`): Queries, each with a short description as key to the `dict`, to
          construct tables in the `queries` type.
        - `query_args` (`Optional[Dict[str, Dict[str, Any]]]`): `kwargs` to
          [`Database.query`](../schema/database/base#irg.schema.database.base.Database.query) per query if needed.
          Keys of the `dict` should match the keys of `queries` if provided.
        - `save_eval_res_to` (`Optional[str]`): Path to save extra evaluation result that is not returned.
          Not saved if not provided.
        - `save_tables_to` (`Optional[str]`): Path to save the constructed tabular data in the format of `Table` based
          on the real database.
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
        self._all_fk = ([] if parent_child_pairs is None else parent_child_pairs) \
                       + (real.foreign_keys if all_direct_parent_child else [])
        self._queries = queries if queries is not None else {}
        self._query_args = query_args if isinstance(query_args, DefaultDict) else defaultdict(dict, query_args) \
            if query_args is not None else defaultdict(dict)

        self._real_tables = self._construct_tables(real)
        if save_tables_to is not None:
            os.makedirs(save_tables_to, exist_ok=True)

        eval_args = defaultdict(lambda: defaultdict(lambda: default_args))
        for type_descr, tables_in_type in tabular_args.items():
            for table_descr, table_args in tables_in_type.items():
                eval_args[type_descr][table_descr] |= table_args
        self._evaluators = {}
        if save_eval_res_to is not None:
            os.makedirs(save_eval_res_to, exist_ok=True)
        for type_descr, tables_in_type in self._real_tables.items():
            evaluators = {}
            if save_eval_res_to is not None:
                os.makedirs(os.path.join(save_eval_res_to, type_descr), exist_ok=True)
            if save_tables_to is not None:
                os.makedirs(os.path.join(save_tables_to, type_descr), exist_ok=True)
            for table_descr, table in tables_in_type.items():
                evaluator = SyntheticTableEvaluator(
                    save_to=os.path.join(save_eval_res_to, type_descr, table_descr)
                    if save_eval_res_to is not None else None,
                    **eval_args[type_descr][table_descr]
                )
                evaluators[table_descr] = evaluator
                if save_tables_to is not None:
                    table.save(os.path.join(save_tables_to, type_descr, f'{table_descr}.pkl'))
            self._evaluators[type_descr] = evaluators

    def _construct_tables(self, db: Database) -> Dict[str, Dict[str, Table]]:
        result = {}

        if self._eval_tables:
            result['tables'] = {table: db[table] for table in self._tables}.items() \
                if self._tables is not None else db.tables

        if self._eval_parent_child:
            result['parent child'] = {
                f'{i} : {fk.child} : {fk.parent}': db.join(fk) for i, fk in enumerate(self._all_fk)
            }

        if self._eval_joined:
            result['joined'] = {'': db.all_joined}

        if self._eval_queries:
            result['queries'] = {
                descr: db.query(query, descr, **self._query_args[descr])
                for descr, query in self._queries
            }

        return result

    def evaluate(self, synthetic: SyntheticDatabase, mean: str = 'arithmetic', smooth: float = 0.1,
                 save_complete_result_to: Optional[str] = None, save_synthetic_tables_to: Optional[str] = None) \
            -> pd.DataFrame:
        """
        Evaluate synthetic database.

        **Args**:

        - `synthetic` (`SyntheticDatabase`): Synthetic database.
        - `mean` and `smooth`: Arguments to
          [`SyntheticTableEvaluator.summary`](./tabular#irg.metrics.tabular.SyntheticTableEvaluator.summary).
        - `save_complete_result_to` (`Optional[str]`): Path to save complete evaluation results for each tabular data
          set to. If it is not provided, this piece of information is not saved.
        - `save_synthetic_tables_to` (`Optional[str]`): Path to save constructed tabular data set of the synthetic
          database.

        **Return**: A `pd.DataFrame` describing the metrics result.
        """
        synthetic_tables = self._construct_tables(synthetic)
        if save_synthetic_tables_to is not None:
            os.makedirs(save_synthetic_tables_to, exist_ok=True)
        results, summary = {}, {}
        for type_descr, evaluators_in_type in self._evaluators.items():
            type_results, type_summary = {}, {}
            if save_synthetic_tables_to is not None:
                os.makedirs(os.path.join(save_synthetic_tables_to, type_descr), exist_ok=True)
            for table_descr, evaluator in evaluators_in_type.items():
                evaluator.evaluate(
                    self._real_tables[type_descr][table_descr],
                    synthetic_tables[type_descr][table_descr]
                )
                type_results[table_descr] = evaluator.result
                type_summary[table_descr] = evaluator.summary(mean, smooth)
                if save_synthetic_tables_to is not None:
                    synthetic_tables[type_descr][table_descr] \
                        .save(os.path.join(save_synthetic_tables_to, type_descr, f'{table_descr}.pkl'))
            results[type_descr] = type_results
            summary[type_descr] = pd.DataFrame(type_summary)

        if save_complete_result_to is not None:
            with open(save_complete_result_to, 'wb') as f:
                pickle.dump(results, f)
        return pd.concat(summary, axis=1)
