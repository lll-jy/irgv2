"""Model trainers for tabular data generation."""

from typing import Dict

from .base import TabularTrainer
from .ctgan import CTGANTrainer
from .tvae import TVAETrainer
from .mlp import MLPTrainer


__all__ = (
    'TabularTrainer',
    'CTGANTrainer',
    'TVAETrainer',
    'MLPTrainer',
    'create_trainer'
)

_TAB_TRAINERS: Dict[str, TabularTrainer.__class__] = {
    'CTGAN': CTGANTrainer,
    'TVAE': TVAETrainer,
    'MLP': MLPTrainer,
}


def create_trainer(trainer_type: str = 'CTGAN', **kwargs) -> TabularTrainer:
    """
    Create trainer of the specified type.

    **Args**:

    - `trainer_type` (`str`): The trainer type, can be 'CTGAN' or 'TVAE'.
    - `kwargs`: Arguments to the corresponding trainer type constructor.
    """
    return _TAB_TRAINERS[trainer_type](**kwargs)
