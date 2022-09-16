"""
ALSET data processing.

ALSET stands for NUS Applied Learning Sciences and Educational Technology data lake,
with many different tables of current and past NUS students.

Data content of ALSET is not public.
"""

from .processor import ALSET_PROCESSORS, ALSET_META_CONSTRUCTORS, ALSET_PROCESS_NAME_MAP, ALSETProcessor

__all__ = (
    'ALSET_PROCESSORS',
    'ALSET_META_CONSTRUCTORS',
    'ALSET_PROCESS_NAME_MAP',
    'ALSETProcessor'
)
