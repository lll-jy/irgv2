"""Database joining mechanism as if all tables are unrelated."""

from typing import Any

from .base import Database


class UnrelatedDatabase(Database):
    """
    Database with joining mechanism as if all tables are unrelated.
    """
    @property
    def mtype(self) -> str:
        return 'unrelated'

    def augment(self):
        pass

    @staticmethod
    def _update_cls(item: Any):
        item.__class__ = UnrelatedDatabase
