"""Table data structure that holds data and metadata of tables in a database."""

from .table import Table, SyntheticTable
from .series import SeriesTable

__all__ = (
    'Table',
    'SyntheticTable',
    'SeriesTable'
)
