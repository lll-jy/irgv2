"""
Metadata of the tables.
"""

from .region import country, provstate, city
from .events import events, life_damage, eco_damage, hostkid, info_int, attack_type, target

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
    'target'
)
