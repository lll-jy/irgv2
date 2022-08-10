from typing import Tuple, Dict, Optional

from torch import Tensor

from ..utils import Trainer, InferenceOutput


class MLPTrainer(Trainer):
    def __init__(self):
        pass

    def _reload_checkpoint(self, idx: int, by: str):
        pass

    def _save_checkpoint(self, idx: int, by: str):
        pass

    def run_step(self, known: Tensor, unknown: Tensor) -> Tuple[Dict[str, float], Optional[Tensor]]:
        pass

    def inference(self, known: Tensor, batch_size: int) -> InferenceOutput:
        pass
