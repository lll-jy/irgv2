"""Schema-related structures, including attributes, tables, and databases."""

from .attribute import BaseAttribute as Attribute, create as create_attribute
from .table import Table, SyntheticTable
from .database import Database, SyntheticDatabase, create as create_db


__all__ = (
    'Attribute',
    'Table',
    'SyntheticTable',
    'Database',
    'SyntheticDatabase',
    'create_attribute',
    'create_db'
)
