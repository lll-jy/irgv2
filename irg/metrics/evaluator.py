"""Evaluator for synthetic database generation."""

from collections import defaultdict
from typing import Optional, List, Dict, Any
import os

from ..schema import Database, SyntheticDatabase
from ..schema.database.base import ForeignKey
from .tabular import SyntheticTableEvaluator


class SyntheticDatabaseEvaluator:
    """Evaluator for synthetic database generation."""
    def __init__(self, real: Database, tables: Optional[List[str]] = None,
                 parent_child_pairs: Optional[List[ForeignKey]] = None, all_direct_parent_child: bool = True,
                 queries: Optional[Dict[str, str]] = None, n_random_queries: int = 10, all_joined: bool = True,
                 save_eval_res_to: Optional[str] = None, save_tables_to: Optional[str] = None,
                 tabular_args: Optional[Dict[str, Dict[str, Dict]]] = None,
                 default_args: Optional[Dict[str, Any]] = None):
        tables = {table: real[table] for table in tables}.items() \
            if tables is not None else real.tables

        all_fk = ([] if parent_child_pairs is None else parent_child_pairs)\
                 + (real.foreign_keys if all_direct_parent_child else [])
        parent_child_tables = {
            f'{fk.child} : {fk.parent}': real.join(fk) for fk in all_fk
        }

        self._tables = {
            'tables': tables,
            'parent child': parent_child_tables
        }
        if save_tables_to is not None:
            os.makedirs(save_tables_to, exist_ok=True)

        eval_args = defaultdict(lambda: defaultdict(lambda: default_args))
        for type_descr, tables_in_type in tabular_args.items():
            for table_descr, table_args in tables_in_type.items():
                eval_args[type_descr][table_descr] |= table_args
        self._evaluators = {}
        if save_eval_res_to is not None:
            os.makedirs(save_eval_res_to, exist_ok=True)
        for type_descr, tables_in_type in self._tables.items():
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

    def evaluate(self, synthetic: SyntheticDatabase):
        pass
