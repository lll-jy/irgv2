from typing import Tuple, Dict, Optional

from torch import Tensor

from ..utils import Trainer


class CTGANTrainer(Trainer):
    def __init__(self):
        pass

    def train(self, known: Tensor, unknown: Tensor, epochs: int, batch_size: int, shuffle: bool = True,
              save_freq: int = 100):
        pass

    def _save_checkpoint(self, idx: int, by: str):
        pass

    def run_step(self, known: Tensor, unknown: Tensor) -> Tuple[Dict[str: float], Optional[Tensor]]:
        pass

    def calculate_loss(self, pred: Tensor, target: Tensor):
        pass