"""Tabular metrics."""

from abc import ABC, abstractmethod
from typing import List, Optional
from statistics import harmonic_mean
import os

import pandas as pd
import numpy as np
from sdv.evaluation import evaluate as sdv_evaluate
import matplotlib.pyplot as plt

from ...schema import Table, SyntheticTable


class BaseMetric(ABC):
    """Tabular metrics."""
    @abstractmethod
    def evaluate(self, real: Table, synthetic: SyntheticTable) -> pd.Series:
        """
        Evaluate one pair of real and synthetic table.

        **Args**:

        - `real` (`Table`): Real table.
        - `synthetic` (`Table`): Synthetic table.

        **Return**: Result of the metric described as a series.
        """
        raise NotImplementedError()


class StatsMetric(BaseMetric):
    """Metric using [SDV](https://sdv.dev/SDV/user_guides/evaluation/single_table_metrics.html) statistical metrics."""
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        **Args**:

        - `metrics` (`Optional[List[str]]`): Metric names (`metrics` argument for `sdv.evaluation.evaluate`).
          Default is ['CSTest', 'KSTest', 'BNLogLikelihood', 'GMLogLikelihood'].
        """
        self._metrics = metrics if metrics is not None \
            else ['CSTest', 'KSTest', 'BNLogLikelihood', 'GMLogLikelihood']

    def evaluate(self, real: Table, synthetic: SyntheticTable) -> pd.Series:
        real_data = real.data(with_id='none').copy()
        synthetic_data = synthetic.data(with_id='none').copy()

        for name, attr in real.attributes.items():
            if attr.atype == 'categorical':
                real_data[name] = real_data[name].apply(lambda x: f'c{x}')
                synthetic_data[name] = synthetic_data[name].apply(lambda x: f'c{x}')
            elif attr.atype == 'datetime':
                real_data[name] = real_data[name] \
                    .apply(lambda x: np.nan if pd.isnull(x) else x.toordinal()).astype('float32')
                synthetic_data[name] = synthetic_data[name] \
                    .apply(lambda x: np.nan if pd.isnull(x) else x.toordinal()).astype('float32')

        eval_res = sdv_evaluate(synthetic_data, real_data, metrics=self._stat_metrics, aggregate=False)
        res = pd.Series()
        for i, row in eval_res.iterrows():
            res.loc[row['metric']] = row['raw_score']
        return res


class CorrMatMetric(BaseMetric):
    """
    Metric inspecting the correlation matrices.
    The metric values are 1 minus the (absolute) differences of correlation matrices between real and fake tables,
    averaged over columns.
    The range of the metric values is [0, 1], the larger the better.
    This metric is calculated using the normalized data.
    """
    def __init__(self, mean: str = 'arithmetic', smooth: float = 0.1, save_to: Optional[str] = None):
        """
        **Args**:

        - `mean` (`str`): The way mean is calculated. Can be either 'arithmetic' or 'harmonic'. Default is 'arithmetic'.
        - `smooth` (`float`): Smoothing value in case of zero division when calculating harmonic mean.
          The harmonic value calculated will be HM(x + smooth) - smooth, and when smooth = 0, it is not smoothed.
        - `save_to` (`Optional[str]`): Path of the directory to save the correlation graphs. If `None`, it is not saved.
          The file saved will have name of the table.
        """
        if mean not in {'arithmetic', 'harmonic'}:
            raise NotImplementedError(f'Mean by {mean} is not implemented. Please give "arithmetic" or "harmonic".')
        if not 0 <= smooth <= 1:
            raise ValueError(f'Smooth value should be in [0, 1], got {smooth}.')
        self._mean, self._smooth, self._save_to = mean, smooth, save_to

    def evaluate(self, real: Table, synthetic: SyntheticTable) -> pd.Series:
        real_data = real.data(normalize=True, with_id='none')
        synthetic_data = synthetic.data(normalize=True, with_id='none')
        r_corr, s_corr = real_data.corr(), synthetic_data.corr()
        diff_corr = 1 - (r_corr - s_corr).abs()

        if self._mean == 'arithmetic':
            res = diff_corr.aggregate('mean')
        else:
            res = diff_corr.aggregate(lambda x: harmonic_mean(x + self._smooth) - self._smooth)

        if self._save_to is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle(f'Correlation matrices of {real.name}')

            def _draw_corr(ax: plt.Axes, name: str, data: pd.DataFrame):
                data[':dummy1'] = -1
                data[':dummy2'] = 1
                ax.matshow(data)
                ax.set_title(name)
                a, b = ax.axes.get_xlim()
                ax.set_xlim(a, b-2)
            _draw_corr(ax1, 'real', r_corr)
            _draw_corr(ax2, 'fake', s_corr)
            plt.savefig(os.path.join(self._save_to, f'{real.name}.png'))

        return res
