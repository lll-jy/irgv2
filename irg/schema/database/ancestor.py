"""
Ancestor-descendant augmenting mechanism.
When generating a table, only tables that this table directly or indirectly references are taken into consideration.
For example, if T1 is referenced by T2 and T3, which are the only foreign keys in the database,
then when generating T2 and T3, they are not aware of the existence nor content of each other.
"""
from typing import Any

import pandas as pd
from torch import Tensor

from ..table import Table
from .base import Database, SyntheticDatabase


class AncestorDescendantDatabase(Database):
    """
    Database with joining mechanism involving direct or indirect references.
    """
    def __init__(self, **kwargs):
        """
        **Args**:

        - `kwargs`: Arguments for [`Database`](./base#irg.schema.database.base.Database).
        """
        super().__init__(**kwargs)

    @property
    def mtype(self) -> str:
        return 'ancestor-descendant'

    def augment(self):
        for name, table in self.tables:
            table = Table.load(table)
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
            data, new_ids, new_attr = parent_table.augmented_for_join

            data = pd.concat({prefix: data}, axis=1)
            left = foreign_key.left
            right = [(prefix, col) for _, col in foreign_key.right]
            augmented = augmented.merge(data, how='left', left_on=left, right_on=right)
            degree = degree.merge(data, how='outer', left_on=left, right_on=right)

            id_cols |= {(prefix, col) for col in new_ids}
            attributes |= {(prefix, name): attr for name, attr in new_attr.items()}
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
        item.__class__ = AncestorDescendantDatabase


class SyntheticAncestorDescendantDatabase(AncestorDescendantDatabase, SyntheticDatabase):
    """
    Synthetic database for ancestor-descendant augmenting mechanism.
    """
    def degree_known_for(self, table_name: str) -> Tensor:
        self._augment_table(table_name, self[table_name])
        known, _, _ = self[table_name].deg_data
        return known
