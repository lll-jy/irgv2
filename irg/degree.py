import json
import os

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBRegressor

from .utils import placeholder


def train_degrees(context: np.ndarray, degrees: np.ndarray, model_dir: str, **kwargs):
    os.makedirs(model_dir, exist_ok=True)

    regressor = XGBRegressor(**kwargs)
    regressor.fit(context, degrees)
    torch.save(regressor, os.path.join(model_dir, "reg.pt"))

    non_zero = degrees > 0
    non_zero_X = context[non_zero]
    non_zero_y = degrees[non_zero]
    pred = regressor.predict(non_zero_X)
    err = np.abs(pred - non_zero_y) / degrees.max()
    zero_ratio = 1 - np.mean(non_zero)
    threshold = np.percentile(err, 95)

    qt = QuantileTransformer(n_quantiles=1000, output_distribution="uniform", ignore_implicit_zeros=False)
    qt.fit(non_zero_y.reshape((-1, 1)))
    torch.save(qt, os.path.join(model_dir, "qt.pt"))

    with open(os.path.join(model_dir, "info.json"), "w") as f:
        json.dump({
            "zero ratio": zero_ratio, "sum deg": int(non_zero_y.sum()), "threshold": threshold
        }, f)


@placeholder
def predict_degrees(context: np.ndarray, model_dir: str, expected_sum: int, tolerance: float = 0.9,
                    min_val: int = 0, max_val: int = np.inf) -> np.ndarray:
    print("In actual implementation, additional logic or advanced logic is required to handle the numerical error.")
    with open(os.path.join(model_dir, "info.json"), "r") as f:
        info = json.load(f)
    zero_ratio = info["zero ratio"]
    real_sum = info["sum deg"]
    quantile_weight = info["threshold"]

    regressor = torch.load(os.path.join(model_dir, "reg.pt"))
    predicted_degrees = regressor.predict(context)
    predicted_degrees += np.random.normal(0, 1e-3, size=predicted_degrees.shape)  # avoid too many equal

    predicted_qt = QuantileTransformer(n_quantiles=1000, output_distribution="uniform", ignore_implicit_zeros=False)
    predicted_quantiles = predicted_qt.fit_transform(predicted_degrees.reshape((-1, 1)))[:, 0]
    predicted_quantiles += np.random.normal(0, 1e-3, size=predicted_quantiles.shape)  # avoid too many equal
    is_zero = predicted_quantiles <= zero_ratio
    actual_qt = torch.load(os.path.join(model_dir, "qt.pt"))
    recovered_non_zero_degrees = actual_qt.inverse_transform(predicted_quantiles[~is_zero].reshape((-1, 1)))[:, 0]
    scaled_degrees = np.zeros_like(predicted_degrees)
    scaled_degrees[~is_zero] = recovered_non_zero_degrees * expected_sum / real_sum

    weighted_degrees = scaled_degrees * quantile_weight + predicted_degrees * (1 - quantile_weight)
    weighted_degrees = np.clip(weighted_degrees, min_val, max_val)
    min_sum = expected_sum * (1 - tolerance)
    max_sum = expected_sum * (1 + tolerance)
    ceil = np.ceil(weighted_degrees).astype("int")
    floor = np.floor(weighted_degrees).astype("int")
    ceil_sum = ceil.sum()
    floor_sum = floor.sum()
    if ceil_sum < min_sum:
        weighted_degrees = ceil
        return _augment_degrees(ceil_sum, min_sum, max_val, weighted_degrees)
    elif floor_sum > max_sum:
        weighted_degrees = floor
        return _reduce_degrees(floor_sum, max_sum, min_val, weighted_degrees)
    else:
        left_threshold = 0
        right_threshold = 1
        cnt = 0
        degrees = np.round(weighted_degrees)
        while right_threshold > left_threshold:
            if cnt > 20:
                break
            threshold = (left_threshold + right_threshold) / 2
            degrees = np.floor(weighted_degrees + threshold)
            sum_degrees = degrees.sum()
            if min_sum <= sum_degrees <= max_sum:
                return degrees
            elif sum_degrees < min_sum:
                left_threshold = threshold
            else:
                right_threshold = threshold
            cnt += 1

        sum_degrees = degrees.sum()
        if min_sum <= sum_degrees <= max_sum:
            return degrees
        elif sum_degrees > max_sum:
            return _reduce_degrees(sum_degrees, max_sum, min_val, weighted_degrees)
        else:
            return _augment_degrees(sum_degrees, min_sum, max_val, weighted_degrees)


def _augment_degrees(current_sum: int, min_sum: float, max_val: int, current_degrees: np.ndarray) -> np.ndarray:
    n_to_add = int(np.ceil(min_sum - current_sum))
    if np.isfinite(max_val):
        to_add = np.random.choice(
            np.repeat(np.arange(current_degrees.shape[0]), max_val - current_degrees), n_to_add, replace=False
        )
    else:
        to_add = np.random.randint(0, current_degrees.shape[0], size=n_to_add)
    vals, cnts = np.unique(to_add, return_counts=True)
    current_degrees[vals] += cnts
    return current_degrees


def _reduce_degrees(current_sum: int, max_sum: float, min_val: int, current_degrees: np.ndarray) -> np.ndarray:
    n_to_reduce = int(np.floor(current_sum - max_sum))
    to_reduce = np.random.choice(
        np.repeat(np.arange(current_degrees.shape[0]), current_degrees - min_val), n_to_reduce, replace=False
    )
    vals, cnts = np.unique(to_reduce, return_counts=True)
    current_degrees[vals] -= cnts
    return current_degrees
