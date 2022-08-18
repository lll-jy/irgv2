"""Handler for numerical data."""
import pickle
import os
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import BayesianGaussianMixture

from .base import BaseAttribute, BaseTransformer
from .categorical import CategoricalTransformer
from ...utils.io import pd_to_pickle


# GMM part adatped from https://github.com/sdv-dev/RDT/blob/stable/rdt/transformers/numerical.py
class NumericalTransformer(BaseTransformer):
    """
    Transformer for numerical data.

    The transformed columns (after `is_nan`) are `value`, `cluster_0`, `cluster_1`, ..., `cluster_k`
    for an attribute with k+1 clusters.
    """
    def __init__(self, temp_cache: str = '.temp', rounding: Optional[int] = None,
                 min_val: float = -np.inf, max_val: float = np.inf,
                 max_clusters: int = 10, std_multiplier: int = 4, weight_threshold: float = 0.005):
        """
        **Args**:

        - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.
        - `rounding` (`Optional[int]`) [default `None`]: Argument to pass to the second argument of `round` function
          is rounding is needed.
          That is, for data generated, output `round(x, rounding)`. If `None`, the data is not rounded.
        - `min_val` (`float`) [default -inf]: The lower bound of the data.
        - `max_val` (`float`) [default inf]: The upper bound of the data.
        - `max_clusters` (`int`) [default 10]: The data is fitted using Gaussian Mixture model, and this parameter
          sets the maximum clusters.
        - `std_multiplier` (`int`) [default 4]: The standard deviation multipler to GMM.
        - `weight_threshold` (`float`) [default 0.005]: The minimum weight for retaining a cluster.
        """
        super().__init__(temp_cache)
        self._rounding, self._min_val, self._max_val = rounding, min_val, max_val
        self._minmax_scaler: Optional[MinMaxScaler] = None

        self._clusters, self._max_clusters = 0, max_clusters
        self._std_multiplier, self._weight_threshold = std_multiplier, weight_threshold

        self._bgm_transformer: Optional[BayesianGaussianMixture] = None
        self._valid_component_indicator = None
        self._component_indicator_transformer = CategoricalTransformer()

    def _load_additional_info(self):
        if os.path.exists(os.path.join(self._temp_cache, 'info.pkl')):
            with open(os.path.join(self._temp_cache, 'info.pkl'), 'rb') as f:
                loaded = pickle.load(f)
            self._minmax_scaler, self._bgm_transformer = loaded['scaler'], loaded['bgm']
        else:
            self._minmax_scaler = MinMaxScaler()
            self._bgm_transformer = BayesianGaussianMixture(
                n_components=self._max_clusters,
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=0.001,
                n_init=1
            )

    def _unload_additional_info(self):
        self._minmax_scaler, self._bgm_transformer = None, None

    def _save_additional_info(self):
        with open(os.path.join(self._temp_cache, 'info.pkl'), 'wb') as f:
            pickle.dump({
                'scaler': self._minmax_scaler,
                'bgm': self._bgm_transformer
            }, f)

    @property
    def atype(self) -> str:
        return 'numerical'

    def _calc_dim(self) -> int:
        return self._clusters + 1

    def _calc_fill_nan(self, original: pd.Series) -> float:
        val = original.mean()
        if pd.isnull(val):
            return 0
        return val

    def _fit(self, original: pd.Series, nan_info: pd.DataFrame):
        minmax_transformed = self._minmax_scaler.fit_transform(nan_info['original'].to_numpy().reshape(-1, 1))

        self._bgm_transformer.fit(minmax_transformed)
        self._valid_component_indicator = self._bgm_transformer.weights_ > self._weight_threshold

        transformed = self._transform(nan_info)
        self._transformed_columns = transformed.columns
        pd_to_pickle(transformed, self._transformed_path)

    def _transform(self, nan_info: pd.DataFrame) -> pd.DataFrame:
        scaled = self._minmax_scaler.transform(nan_info['original'].to_numpy().reshape(-1, 1))
        means = self._bgm_transformer.means_.reshape((1, self._max_clusters))

        stds = np.sqrt(self._bgm_transformer.covariances_).reshape((1, self._max_clusters))
        normalized_values = (scaled - means) / (self._std_multiplier * stds)
        normalized_values = normalized_values[:, self._valid_component_indicator]
        component_probs = self._bgm_transformer.predict_proba(scaled)
        component_probs = component_probs[:, self._valid_component_indicator]

        selected_component = np.zeros(len(scaled), dtype='int')
        for i in range(len(scaled)):
            component_prob_t = component_probs[i] + 1e-6
            component_prob_t = component_prob_t / component_prob_t.sum()
            selected_component[i] = np.random.choice(
                np.arange(self._valid_component_indicator.sum()),
                p=component_prob_t
            )

        aranged = np.arange(len(scaled))
        normalized = normalized_values[aranged, selected_component].reshape([-1, 1])
        normalized = normalized[:, 0]

        selected_component = pd.Series(selected_component, dtype='str')
        if self._fitted:
            selected_component = self._component_indicator_transformer.transform(selected_component).iloc[:, 1:]
        else:
            self._clusters = len(set(selected_component))
            self._component_indicator_transformer.fit(selected_component)
            selected_component = self._component_indicator_transformer.get_original_transformed().iloc[:, 1:]

        rows = [normalized.reshape(len(normalized), 1), selected_component.to_numpy()]
        col_names = ['value'] + [f'cluster_{i}' for i in range(self._clusters)]
        result = pd.DataFrame(np.hstack(rows), columns=col_names)
        result.insert(0, 'is_nan', nan_info['is_nan'])
        return result.fillna(0).astype('float32')

    def _categorical_dimensions(self) -> List[Tuple[int, int]]:
        return [(0, 1), (2, self._clusters+2)]

    def _inverse_transform(self, data: pd.DataFrame) -> pd.Series:
        normalized = data[:, 0]
        means = self._bgm_transformer.means_.reshape([-1])
        stds = np.sqrt(self._bgm_transformer.covariances_).reshape([-1])

        selected_component = data[:, 1:]
        col_names = [f'cat{i}' for i in range(self._clusters)]
        selected_component = pd.DataFrame(selected_component, columns=col_names)
        selected_component = self._component_indicator_transformer \
            .inverse_transform(selected_component).astype(int)

        std_t = stds[self._valid_component_indicator][selected_component]
        mean_t = means[self._valid_component_indicator][selected_component]
        reversed_data = normalized * self._std_multiplier * std_t + mean_t

        reversed_data = self._minmax_scaler \
            .inverse_transform(reversed_data.reshape(len(reversed_data), 1))
        reversed_data = pd.Series(reversed_data[:, 0])
        return reversed_data.apply(self._round_minmax)

    def _round_minmax(self, v):
        if self._rounding is not None:
            v = round(v, self._rounding)
        v = max(self._min_val, v)
        v = min(self._max_val, v)
        return v


class NumericalAttribute(BaseAttribute):
    """Attribute for numerical data."""
    def __init__(self, name: str, values: Optional[pd.Series] = None, temp_cache: str = '.temp', **kwargs):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        - `temp_cache` (`str`): Directory path to save cached temporary files. Default is `.temp`.
        - `kwargs`: Arguments for `NumericalTransformer`.
        """
        self._kwargs = kwargs
        super().__init__(name, 'numerical', values, temp_cache)

    def _create_transformer(self):
        self._transformer = NumericalTransformer(self._temp_cache, **self._kwargs)

    def __copy__(self) -> "NumericalAttribute":
        new_attr = super().__copy__()
        new_attr.__class__ = NumericalAttribute
        new_attr._kwargs = self._kwargs
        return new_attr
