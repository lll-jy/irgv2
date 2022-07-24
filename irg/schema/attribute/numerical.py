"""Handler for numerical data."""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import BayesianGaussianMixture

from .base import BaseAttribute, BaseTransformer
from .categorical import CategoricalTransformer


# GMM part adatped from https://github.com/sdv-dev/RDT/blob/stable/rdt/transformers/numerical.py
class NumericalTransformer(BaseTransformer):
    """Transformer for numerical data."""
    def __init__(self, rounding: Optional[int] = None, min_val: float = -np.inf, max_val: float = np.inf,
                 max_clusters: int = 10, std_multiplier: int = 4, weight_threshold: float = 0.005):
        """
        **Args**:

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
        super().__init__()
        self._rounding, self._min_val, self._max_val = rounding, min_val, max_val
        self._minmax_scaler = MinMaxScaler()

        self._clusters, self._max_clusters = 0, max_clusters
        self._std_multiplier, self._weight_threshold = std_multiplier, weight_threshold

        self._bgm_transformer = BayesianGaussianMixture(
            n_components=self._max_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            n_init=1
        )
        self._valid_component_indicator = None
        self._component_indicator_transformer = CategoricalTransformer()

    @property
    def atype(self) -> str:
        return 'numerical'

    def _calc_dim(self) -> int:
        return self._clusters + 1

    def _calc_fill_nan(self) -> float:
        val = self._original.mean()
        if pd.isnull(val):
            return 0
        return val

    def _fit(self):
        minmax_transformed = self._minmax_scaler.fit_transform(self._original.to_numpy().reshape(-1, 1))

        self._bgm_transformer.fit(minmax_transformed)
        self._valid_component_indicator = self._bgm_transformer.weights_ > self._weight_threshold

        self._transformed = self._transform(self._nan_info)

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
            selected_component = self._component_indicator_transformer.transform(selected_component)
        else:
            self._clusters = len(set(selected_component))
            self._component_indicator_transformer.fit(selected_component)
            selected_component = self._component_indicator_transformer.get_original_transformed()

        rows = [normalized.reshape(len(normalized), 1), selected_component.to_numpy()]
        col_names = ['value'] + [f'cluster_{i}' for i in range(self._clusters)]
        return pd.DataFrame(np.hstack(rows), columns=col_names)

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
    def __init__(self, name: str, values: Optional[pd.Series] = None, **kwargs):
        """
        **Args**:

        - `name` (`str`): Name of the attribute.
        - `values` (`Optional[pd.Series]`): Data of the attribute (that is used for fitting normalization transformers).
        - `kwargs`: Arguments for `NumericalTransformer`.
        """
        super().__init__(name, 'numerical', values)
        self._kwargs = kwargs

    @property
    def atype(self) -> str:
        return 'numerical'

    def _create_transformer(self):
        self._transformer = NumericalTransformer(**self._kwargs)
