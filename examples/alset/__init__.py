"""
ALSET data processing.

ALSET stands for NUS Applied Learning Sciences and Educational Technology data lake,
with many different tables of current and past NUS students.

Data content of ALSET is not public.
"""

from typing import Dict
from types import FunctionType

from . import data

__all__ = (
    'ALSET_PROCESSORS',
)

ALSET_PROCESSORS: Dict[str, FunctionType] = {
    'personal_data': data.personal_data
}
"""ALSET table data processors."""
