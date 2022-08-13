"""
Examples of processing databases for training/generation preparation.

Example databases include:

- NUS ALSET Data: NUS Applied Learning Sciences and Educational Technology data lake (not public).
- Global Terrorism Database: Information on more than 180,000 Terrorist Attacks (available on Kaggle).
"""

from typing import Dict
from types import FunctionType

from .alset import ALSET_PROCESSORS
from .rtd import RTD_PROCESSORS, RTD_META_CONSTRUCTORS

__all__ = (
    'PROCESSORS',
    'META_CONSTRUCTORS'
)

PROCESSORS: Dict[str, Dict[str, FunctionType]] = {
    'alset': ALSET_PROCESSORS,
    'rtd': RTD_PROCESSORS
}
"""All table data processorss."""

META_CONSTRUCTORS: Dict[str, Dict[str, FunctionType]] = {
    'alset': None,
    'rtd': RTD_META_CONSTRUCTORS
}
"""All constructors for metadata of tables."""
