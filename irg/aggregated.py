import numpy as np

from .utils import placeholder, train_mlp, predict_mlp


@placeholder
def train_aggregated_information(context: np.ndarray, aggregated_info: np.ndarray, model_dir: str, **kwargs):
    print("We use a simple MLP predictor in place, but actually we adapted TAEGAN.")
    train_mlp(context, aggregated_info, model_dir, **kwargs)


@placeholder
def generate_aggregated_information(context: np.ndarray, model_dir: str) -> np.ndarray:
    print("We use a simple MLP predictor in place, but actually we adapted TAEGAN.")
    return predict_mlp(context, model_dir)
