import os

import numpy as np
import pandas as pd
from ctgan import CTGAN

from .utils import placeholder


@placeholder
def train_standalone(data: np.ndarray, model_dir: str, **kwargs):
    print("Training standalone tabular model can be replaced by other models. In original IRG, we use TAEGAN. "
          "Here we use CTGAN.")
    os.makedirs(model_dir, exist_ok=True)
    model = CTGAN(**kwargs)
    data = pd.DataFrame(data, columns=[f"dim{i:02d}" for i in range(data.shape[1])])
    model.fit(data)
    model.save(os.path.join(model_dir, "model.pkl"))


@placeholder
def generate_standalone(n: int, model_dir: str, ) -> np.ndarray:
    print("Generating standalone tabular model can be replaced by other models. In original IRG, we use TAEGAN. "
          "Here we use CTGAN.")
    model = CTGAN.load(os.path.join(model_dir, "model.pkl"))
    generated = model.sample(n)
    return generated.values
