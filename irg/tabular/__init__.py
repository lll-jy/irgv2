"""Model trainers for tabular data generation."""

from typing import Dict

from ..utils import Trainer
from .ctgan import CTGANTrainer
from .tvae import TVAETrainer


__all__ = (
    'CTGANTrainer',
    'TVAETrainer',
    'create_trainer'
)

_TAB_TRAINERS: Dict[str, Trainer.__class__] = {
    'CTGAN': CTGANTrainer,
    'TVAE': TVAETrainer
}


def create_trainer(trainer_type: str = 'CTGAN', **kwargs) -> Trainer:
    """
    Create trainer of the specified type.

    **Args**:

    - `trainer_type` (`str`): The trainer type, can be 'CTGAN' or 'TVAE'.
    - `kwargs`: Arguments to the corresponding trainer type constructor.
    """
    return _TAB_TRAINERS[trainer_type](**kwargs)
