"""
Unrelated augmenting mechanism.
All tables in the database are understood as if there is no relation between any two tables.
"""

from typing import Any, Optional

from torch import Tensor
import torch

from .base import Database, SyntheticDatabase


class UnrelatedDatabase(Database):
    """
    Database with joining mechanism as if all tables are unrelated.
    """
    def __init__(self, **kwargs):
        """
        **Args**:

        - `kwargs`: Arguments for [`Database`](./base#irg.schema.database.base.Database).
        """
        super().__init__(**kwargs)

    @property
    def mtype(self) -> str:
        return 'unrelated'

    def augment(self):
        pass

    @staticmethod
    def _update_cls(item: Any):
        item.__class__ = UnrelatedDatabase


class SyntheticUnrelatedDatabase(UnrelatedDatabase, SyntheticDatabase):
    """
    Synthetic database for unrelated augmenting mechanism.
    """
    def degree_known_for(self, table_name: str) -> Tensor:
        return torch.zeros(0)
