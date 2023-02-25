"""Model trainers for series data generation."""
from typing import Dict

from .base import SeriesTrainer
from .timegan import TimeGANTrainer


__all__ = (
    'SeriesTrainer',
    'TimeGANTrainer',
    'create_trainer'
)

_SER_TRAINERS: Dict[str, SeriesTrainer.__class__] = {
    'TimeGAN': TimeGANTrainer
}


def create_trainer(trainer_type: str = 'TimeGAN', **kwargs) -> SeriesTrainer:
    """
    Create trainer of the specified type.

    **Args**:

    - `trainer_type` (`str`): The trainer type, can be 'TimeGAN'.
    - `kwargs`: Arguments to the corresponding seriestrainer type constructor.
    """
    return _SER_TRAINERS[trainer_type](**kwargs)

