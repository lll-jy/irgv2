"""
Examples of processing databases for training/generation preparation.

Example databases include:

- NUS ALSET Data: NUS Applied Learning Sciences and Educational Technology data lake (not public).
- Global Terrorism Database: Information on more than 180,000 Terrorist Attacks (available on Kaggle).
"""

from typing import Dict
from types import FunctionType

from .processor import DatabaseProcessor
from .adults import AdultsProcessor
from .alset import ALSETProcessor
from .covtype import CovtypeProcessor
from .rtd import RTDProcessor

__all__ = (
    'PROCESSORS',
    'META_CONSTRUCTORS',
    'DatabaseProcessor',
    'DATABASE_PROCESSORS'
)

from .adults import ADULTS_PROCESSORS, ADULTS_META_CONSTRUCTORS, ADULTS_PROCESS_NAME_MAP
from .alset import ALSET_PROCESSORS, ALSET_META_CONSTRUCTORS, ALSET_PROCESS_NAME_MAP
from .covtype import COVTYPE_PROCESSORS, COVTYPE_META_CONSTRUCTORS, COVTYPE_PROCESS_NAME_MAP
from .rtd import RTD_PROCESSORS, RTD_META_CONSTRUCTORS, RTD_PROCESS_NAME_MAP


PROCESSORS: Dict[str, Dict[str, FunctionType]] = {
    'alset': ALSET_PROCESSORS,
    'rtd': RTD_PROCESSORS,
    'adults': ADULTS_PROCESSORS,
    'covtype': COVTYPE_PROCESSORS
}
"""All table data processorss."""

META_CONSTRUCTORS: Dict[str, Dict[str, FunctionType]] = {
    'alset': ALSET_META_CONSTRUCTORS,
    'rtd': RTD_META_CONSTRUCTORS,
    'adults': ADULTS_META_CONSTRUCTORS,
    'covtype' :COVTYPE_META_CONSTRUCTORS
}
"""All constructors for metadata of tables."""

PROCESS_NAME_MAP: Dict[str, Dict[str, str]] = {
    'alset': ALSET_PROCESS_NAME_MAP,
    'rtd': RTD_PROCESS_NAME_MAP,
    'adults': ADULTS_PROCESS_NAME_MAP,
    'covtype': COVTYPE_PROCESS_NAME_MAP
}
"""All source data file names (without extension) for all tables."""

DATABASE_PROCESSORS: Dict[str, DatabaseProcessor.__class__] = {
    'alset': ALSETProcessor,
    'rtd': RTDProcessor,
    'adults': AdultsProcessor,
    'covtype': CovtypeProcessor
}
"""Database processors for all pre-defined databases."""
