"""
Parent-child augmenting mechanism.
When generating a table, only tables that this table directly references are taken into consideration.
For example, if T1 references T2 and T2 references T3, when generating T3, the content of T2 is used but
the content of T1 is left there unused.
"""
from typing import Any

import pandas as pd
from torch import Tensor

from .base import Database, SyntheticDatabase
from ..table import Table


class ParentChildDatabase(Database):
    """
    Database with joining mechanism involving direct parent and children only.
    """
    def __init__(self, **kwargs):
        """
        **Args**:

        - `kwargs`: Arguments for [`Database`](./base#irg.schema.database.base.Database).
        """
        super().__init__(**kwargs)

    @property
    def mtype(self) -> str:
        return 'parent-child'

    def augment(self):
        for name, table in self.tables:
            self._augment_table(name, table)

    def _augment_table(self, name: str, table: Table):
        foreign_keys = self._foreign_keys[name]
        augmented = pd.concat({name: table.data()}, axis=1)
        degree = augmented.copy()
        id_cols, attributes, fk_cols = set(), {}, set()
        for i, foreign_key in enumerate(foreign_keys):
            parent_name = foreign_key.parent
            prefix = f'fk{i}:{parent_name}'
            parent_table = self[parent_name]

            data = pd.concat({prefix: parent_table.data()}, axis=1)
            left = foreign_key.left
            right = [(prefix, col) for _, col in foreign_key.right]
            augmented = augmented.merge(data, how='left', left_on=left, right_on=right)
            degree = degree.merge(data, how='outer', left_on=left, right_on=right)

            id_cols |= {(prefix, col) for col in parent_table.id_cols}
            attributes |= {(prefix, name): attr for name, attr in parent_table.attributes.items()}
            fk_cols |= {col for _, col in foreign_key.left}

        aug_id_cols, deg_id_cols = set(), set()
        aug_attr, deg_attr = {}, {}
        for attr_name, attr in table.attributes.items():
            if attr.atype == 'id':
                aug_id_cols.add((name, attr_name))
                if attr_name in fk_cols:
                    deg_id_cols.add((name, attr_name))
            aug_attr[(name, attr_name)] = attr
            if attr_name in fk_cols:
                deg_attr[(name, attr_name)] = attr
        table.augment(
            augmented=augmented,
            degree=degree.drop(columns=[(name, col) for col in table.columns if col not in fk_cols]),
            augmented_ids=aug_id_cols | id_cols, degree_ids=deg_id_cols | id_cols,
            augmented_attributes=aug_attr | attributes, degree_attributes=deg_attr | attributes
        )

    @staticmethod
    def _update_cls(item: Any):
        item.__class__ = ParentChildDatabase


class SyntheticParentChildDatabase(ParentChildDatabase, SyntheticDatabase):
    """
    Synthetic database for parent-child augmenting mechanism.
    """
    def degree_known_for(self, table_name: str) -> Tensor:
        self._augment_table(table_name, self[table_name])
        known, _, _ = self[table_name].deg_data
        return known
