"""
Table data processing.

The raw data is a joined table of all information.
Hence, all tables are extracted from this joined table.
"""

from .region import country, provstate, city
from .events import events, life_damage, eco_damage, hostkid, info_int, attack_type, target, group, claim, weapon, \
    related

__all__ = (
    'country',
    'provstate',
    'city',
    'events',
    'life_damage',
    'eco_damage',
    'hostkid',
    'info_int',
    'attack_type',
    'target',
    'group',
    'claim',
    'weapon',
    'related'
)
