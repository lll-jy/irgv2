"""
Table data processing.

The raw data is a joined table of all information.
Hence, all tables are extracted from this joined table.
"""

from .region import country, provstate, city

__all__ = (
    'country',
    'provstate',
    'city'
)
