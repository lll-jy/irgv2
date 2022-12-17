"""
Degree generation framework.
"""

from .base import DegreeTrainer
from .ptgan import DegreeAsTabularTrainer
from .neighbors import DegreeFromNeighborsTrainer


__all__ = (
    'DegreeTrainer',
    'DegreeAsTabularTrainer',
    'DegreeFromNeighborsTrainer'
)

_DEG_TRAINERS: Dict[str, DegreeTrainer.__class__] = {
    'as_tab': DegreeAsTabularTrainer,
    'neighbors': DegreeFromNeighborsTrainer
}


def create_trainer(trainer_type: str = 'neighbors', **kwargs) -> DegreeTrainer:
    """
    Create trainer of the specified type.

    **Args**:

    - `trainer_type` (`str`): The trainer type, can be 'as_tab' or 'neighbors'.
    - `kwargs`: Arguments to the corresponding trainer type constructor.
    """
    return _DEG_TRAINERS[trainer_type](**kwargs)
