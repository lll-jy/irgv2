import os

import numpy as np
import torch
from xgboost import XGBClassifier

from .utils import placeholder


@placeholder
def train_isna_indicator(context: np.ndarray, isna: np.ndarray, model_dir: str, **kwargs):
    """
    Train is-N/A indicator model.

    Parameters
    ----------
    context : np.ndarray
        Context for predicting is-N/A indicators.
    isna : np.ndarray
        Boolean indicators of whether each row has N/A foreign key values.
    model_dir : str
        The directory to save the models at.
    **kwargs
        Other parameters regarding classification.
    """
    print("In actual design of IRG, is-N/A ratio should be kept.")
    os.makedirs(model_dir, exist_ok=True)

    classifier = XGBClassifier(**kwargs)
    classifier.fit(context, isna)
    torch.save(classifier, os.path.join(model_dir, "clf.pt"))


@placeholder
def predict_isna_indicator(context: np.ndarray, model_dir: str) -> np.ndarray:
    """
    Predict is-N/A indicators.

    Parameters
    ----------
    context : np.ndarray
        Input context for is-N/A indicator prediction.
    model_dir : str
        The directory where save the models are at.

    Returns
    -------
    np.ndarray
        Predicted is-N/A indicators corresponding to each row in `context`.
    """
    print("In actual design of IRG, values are predicted based on is-N/A ratio.")
    classifier = torch.load(os.path.join(model_dir, "clf.pt"))
    return classifier.predict(context)
