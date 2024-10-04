import numpy as np

from .utils import placeholder, train_mlp, predict_mlp


@placeholder
def train_aggregated_information(context: np.ndarray, aggregated_info: np.ndarray, model_dir: str, **kwargs):
    """
    Train the model for generating aggregated information. This is essentially a multi-variate
    potentially-continuous conditional tabular generation.

    Parameters
    ----------
    context : np.ndarray
        The context to predicted aggregated information for this table.
    aggregated_info : np.ndarray
        The aggregated information for this table.
    model_dir : str
        The directory to save models at.
    **kwargs
        Other parameters for the generator.
    """
    print("We use a simple MLP predictor in place, but actually we adapted TAEGAN.")
    train_mlp(context, aggregated_info, model_dir, **kwargs)


@placeholder
def generate_aggregated_information(context: np.ndarray, model_dir: str) -> np.ndarray:
    """
    Generate aggregated information for a given context.

    Parameters
    ----------
    context : np.ndarray
        The context to predicted aggregated information for this table.
    model_dir : str
        The directory where saved model is.

    Returns
    -------
    np.ndarray
        Generated aggregated information.
    """
    print("We use a simple MLP predictor in place, but actually we adapted TAEGAN.")
    return predict_mlp(context, model_dir)
