import os

import numpy as np
import pandas as pd
from ctgan import CTGAN

from .utils import placeholder


@placeholder
def train_standalone(data: np.ndarray, model_dir: str, **kwargs):
    """
    Train standalone tabular generator.
    
    Parameters
    ----------
    data : np.ndarray
        The encoded data to be generated.
    model_dir : str
        The directory to save trained models.
    **kwargs 
        Other parameters regarding generation.
    """
    print("Training standalone tabular model can be replaced by other models. In original IRG, we use TAEGAN. "
          "Here we use CTGAN.")
    model = CTGAN(**kwargs)
    data = pd.DataFrame(data, columns=[f"dim{i:02d}" for i in range(data.shape[1])])
    model.fit(data)
    model.save(os.path.join(model_dir, "model.pt"))


@placeholder
def generate_standalone(n: int, model_dir: str, ) -> np.ndarray:
    """
    Generate standalone table.

    Parameters
    ----------
    n : int
        The number of rows to generate.
    model_dir : str
        The model directory where trained model is saved.

    Returns
    -------
    np.ndarray
        The generated data before decoding.
    """
    print("Generating standalone tabular model can be replaced by other models. In original IRG, we use TAEGAN. "
          "Here we use CTGAN.")
    model = CTGAN.load(os.path.join(model_dir, "model.pt"))
    generated = model.sample(n)
    return generated.values
