import json
import os.path
from typing import List, Tuple

import numpy as np

from .utils import placeholder, train_mlp, predict_mlp


@placeholder
def train_actual_values(
        context: np.ndarray, length: np.ndarray, values: np.ndarray, groups: List[np.ndarray], model_dir: str, **kwargs
):
    """
    Train the model for generating actual values. This is essentially a multi-variate sequential data generation using
    some multivariate static conditions, and with lengths known.

    Parameters
    ----------
    context: np.ndarray
        The static context to predict values for this table.
    length : np.ndarray
        The lengths of the values to predict.
    values: np.ndarray
        The values to be predicted.
    groups : List[np.ndarray]
        The indices in values for each row in context.
    model_dir: str
        The directory to save the trained model.
    **kwargs
        Other parameters for the generator.
    """
    print("We use a simple non-sequential MLP predictor in place, but actually we adapted TimeVQVAE.")
    x = np.concatenate([context, length.reshape((-1, 1))], axis=1)
    ys = []
    max_length = length.max()
    for i, g in enumerate(groups):
        empty = np.zeros((max_length, values.shape[-1]))
        empty[:length[i]] = values[g]
        ys.append(empty.reshape(-1))
    y = np.stack(ys)
    train_mlp(x, y, model_dir, **kwargs)
    with open(os.path.join(model_dir, "info.json"), "w") as f:
        json.dump({"size": values.shape[-1]}, f)


@placeholder
def generate_actual_values(
        context: np.ndarray, length: np.ndarray, model_dir: str
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate actual values for given static context.

    Parameters
    ----------
    context: np.ndarray
        The static context to predict actual values in the table.
    length : np.ndarray
        The expected lengths per row in context in the output.
    model_dir: str
        The directory where saved model is.

    Returns
    -------
    np.ndarray
        The generated actual values.
    List[np.ndarray]
        The indices per row in context in the output.
    """
    print("We use a simple non-sequential MLP predictor in place, but actually we adapted TimeVQVAE.")
    y = predict_mlp(np.concatenate([context, length.reshape((-1, 1))], axis=1), model_dir)
    with open(os.path.join(model_dir, "info.json"), "r") as f:
        size = json.load(f)["size"]
    values = []
    groups = []
    base_idx = 0
    for l, p in zip(length, y):
        p = p.reshape((-1, size))
        p = p[:int(l)]
        values.append(p)
        groups.append(np.arange(base_idx, base_idx + p.shape[0]))
        base_idx += p.shape[0]

    return np.concatenate(values), groups
