"""Evaluator for synthetic database generation."""

from collections import defaultdict
from typing import Optional, List, Dict, Any
import os
import pickle

import pandas as pd

from ..schema import Database, SyntheticDatabase, Table
from ..schema.database.base import ForeignKey
from .tabular import SyntheticTableEvaluator


class SyntheticDatabaseEvaluator:
    """Evaluator for synthetic database generation."""
    def __init__(self, real: Database,
                 eval_tables: bool = True, eval_parent_child: bool = True, eval_joined: bool = True,
                 eval_queries: bool = True,
                 tables: Optional[List[str]] = None,
                 parent_child_pairs: Optional[List[ForeignKey]] = None, all_direct_parent_child: bool = True,
                 queries: Optional[Dict[str, str]] = None, query_args: Optional[Dict[str, Dict[str, Any]]] = None,
                 save_eval_res_to: Optional[str] = None, save_tables_to: Optional[str] = None,
                 tabular_args: Optional[Dict[str, Dict[str, Dict]]] = None,
                 default_args: Optional[Dict[str, Any]] = None):
        self._real = real
        (self._eval_tables, self._eval_parent_child,
         self._eval_joined, self._eval_queries) = eval_tables, eval_parent_child, eval_joined, eval_queries
        self._tables = tables
        self._all_fk = ([] if parent_child_pairs is None else parent_child_pairs) \
                       + (real.foreign_keys if all_direct_parent_child else [])
        self._queries = queries if queries is not None else {}
        self._query_args = query_args if query_args is not None else defaultdict(dict)

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
                f'{fk.child} : {fk.parent}': db.join(fk) for fk in self._all_fk
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
                    synthetic_tables[type_descr][table_descr]\
                        .save(os.path.join(save_synthetic_tables_to, type_descr, f'{table_descr}.pkl'))
            results[type_descr] = type_results
            summary[type_descr] = pd.DataFrame(type_summary)

        if save_complete_result_to is not None:
            with open(save_complete_result_to, 'wb') as f:
                pickle.dump(results, f)
        return pd.concat(summary, axis=1)
