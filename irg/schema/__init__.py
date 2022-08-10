"""Schema-related structures, including attributes, tables, and databases."""

from .attribute import BaseAttribute as Attribute
from .table import Table
from .database import Database


__all__ = (
    'Attribute',
    'Table',
    'Database'
)
